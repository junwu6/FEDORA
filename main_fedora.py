import logging
import copy
from tqdm import trange
import numpy as np
import torch

from models.ClientUpdare import FLocalUpdateFEDORA
from models.forecasting.SCINet import SCINet
from utils.params import args_parser
from utils.helper import set_seed, set_logger, get_device
from utils.load_data import BaseNodes_load
from evaluation import eval_model


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = torch.sum(adj, dim=1)
    rowsum = torch.flatten(rowsum)
    d_inv_sqrt = torch.pow(rowsum, -1.0)
    d_inv_sqrt[d_inv_sqrt == float("inf")] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(d_mat_inv_sqrt, adj)


def run_FEDORA(args, device):
    nodes = BaseNodes_load(
        args.data_path,
        args.num_nodes,
        batch_size=args.batch_size,
        input_size=args.input_size,
        forecast_horizon=args.forecast_horizon,
        stride=args.stride,
        valid_set_size=args.valid_set_size,
        test_set_size=args.test_set_size,
    )

    g_net = SCINet(
        output_len=args.forecast_horizon,
        input_len=args.input_size,
        input_dim=args.nb_features,
        hid_size=args.hid_size,
        num_stacks=args.num_stacks,
        num_levels=args.num_levels,
        num_decoder_layer=args.num_decoder_layer,
        concat_len=args.concat_len,
        groups=args.groups,
        kernel=args.kernel,
        dropout=args.dropout,
        single_step_output_One=args.single_step_output_One,
        positionalE=args.positionale,
        modified=args.modified,
        RIN=args.RIN,
    )

    net_glob = g_net.to(device)
    net_glob.train()
    w_locals = {}
    w_globals = {}
    for user in range(args.num_nodes):
        w_local_dict = {}
        w_global_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
            w_global_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict
        w_globals[user] = w_global_dict

    last_eval = -1
    step_iter = trange(args.num_steps)

    args.frac = 1.0
    similarity_matrix = torch.zeros(args.num_nodes, args.num_nodes)

    node_subspace = []
    for i in range(args.num_nodes):
        data_loader = nodes.train_loaders[i]
        with torch.no_grad():
            x, y = [], []
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                x.append(data.view(data.shape[0], -1))
                y.append(target)
            x = torch.cat(x, dim=0)
            y = torch.cat(y, dim=0)
        x = torch.cat([x, y.squeeze(-1)], dim=1)
        u_s, s_s, v_s = torch.svd_lowrank(x.t(), q=1)
        node_subspace.append(u_s)

    for i in range(args.num_nodes):
        for j in range(i + 1, args.num_nodes):
            p_s, cospa, p_t = torch.svd(
                torch.mm(node_subspace[i].t(), node_subspace[j])
            )
            sim = torch.norm(cospa, 1)
            sim = torch.exp(sim * 10)
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    # get top-k similarity matrix
    adj = torch.zeros_like(similarity_matrix)
    for i in range(args.num_nodes):
        values, indexes = torch.topk(similarity_matrix[i], 2)
        for neigh in indexes:
            adj[i, neigh] = similarity_matrix[i, neigh]
    similarity_matrix = normalize_adj(adj)
    pa = 0.8
    inv_similarity_matrix = (1 - pa) * torch.linalg.inv(
        torch.eye(args.num_nodes) - pa * similarity_matrix
    )

    for step in step_iter:
        num_selected_nodes = max(int(args.frac * args.num_nodes), 1)
        if step == args.num_steps - 1:
            num_selected_nodes = args.num_nodes
        selected_node_ids = np.random.choice(
            range(args.num_nodes), num_selected_nodes, replace=False
        )

        w_aux = []
        lam_weights = torch.zeros(args.num_nodes)
        for node_id in selected_node_ids:
            client = FLocalUpdateFEDORA(
                args=args,
                train_loader=nodes.train_loaders[node_id],
                val_loader=nodes.val_loaders[node_id],
                node_id=node_id,
            )

            net_global = copy.deepcopy(net_glob)
            w_global = copy.deepcopy(w_globals[node_id])
            net_global.load_state_dict(w_global)
            global_loss = client.get_loss(net=net_global.to(device), device=device)
            w_k, sam_local = client.train(net=net_global.to(device), device=device)
            w_aux.append(w_k)

            net_local = copy.deepcopy(net_glob)
            w_local = copy.deepcopy(w_locals[node_id])
            net_local.load_state_dict(w_local)
            local_loss = client.get_loss(net=net_local.to(device), device=device)
            lam = (local_loss - global_loss).detach()
            if lam < 0:
                lam = 0
            else:
                lam = torch.exp(lam / 2)
            lam_weights[node_id] = lam

            w_local, _ = client.train(
                net=net_local.to(device),
                device=device,
                w_fedora=copy.deepcopy(w_globals[node_id]),
                lam=lam,
            )

            for k, key in enumerate(net_glob.state_dict().keys()):
                w_locals[node_id][key] = w_local[key]
                w_globals[node_id][key] = copy.deepcopy(w_local[key])

        # Server update
        w_globals_old = copy.deepcopy(w_globals)
        w_global_new = copy.deepcopy(w_globals)
        for idx_i in range(len(w_globals)):
            for k in w_global_new[0].keys():
                w_global_new[idx_i][k] = 0
                for idx_j in range(len(w_globals)):
                    w_global_new[idx_i][k] += (
                        inv_similarity_matrix[idx_i, idx_j] * w_globals_old[idx_j][k]
                    )
        w_globals = copy.deepcopy(w_global_new)

        net_locals = []
        for node_id in range(args.num_nodes):
            net_local = copy.deepcopy(net_glob)
            net_local.load_state_dict(w_locals[node_id])
            net_locals.append(net_local)

        step_iter.set_description(f"Step: {step + 1}")
        if step % args.eval_every == 0:
            last_eval = step
            (
                step_results,
                avg_train_loss,
                avg_smape_loss,
                avg_mae_loss,
                avg_mse_loss,
                avg_rmse_loss,
                avg_r2_loss,
            ) = eval_model(
                nodes,
                args.num_nodes,
                net_locals,
                device,
                split="test",
                stacks=args.num_stacks,
            )
            logging.info(
                f"\nStep: {step + 1}, AVG Loss: {avg_train_loss:.4f},  AVG SMAPE: {avg_smape_loss:.4f}, AVG MAE: {avg_mae_loss:.4f}, AVG MSE: {avg_mse_loss:.4f}, AVG RMSE: {avg_rmse_loss:.4f}, AVG R2: {avg_r2_loss:.4f}"
            )

    if step != last_eval:
        (
            step_results,
            avg_train_loss,
            avg_smape_loss,
            avg_mae_loss,
            avg_mse_loss,
            avg_rmse_loss,
            avg_r2_loss,
        ) = eval_model(
            nodes,
            args.num_nodes,
            net_locals,
            device,
            split="test",
            stacks=args.num_stacks,
        )
        logging.info(
            f"\nStep: {step + 1}, AVG Loss: {avg_train_loss:.4f},  AVG SMAPE: {avg_smape_loss:.4f}, AVG MAE: {avg_mae_loss:.4f}, AVG MSE: {avg_mse_loss:.4f}, AVG RMSE: {avg_rmse_loss:.4f}, AVG R2: {avg_r2_loss:.4f}"
        )

    return step_results


if __name__ == "__main__":
    args = args_parser()
    set_logger()
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)
    results = run_FEDORA(args, device)

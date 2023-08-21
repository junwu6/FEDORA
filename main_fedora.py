import logging
import torch
import copy
import numpy as np
from tqdm import trange
from collections import defaultdict
from models.ClientUpdate import LocalUpdateFEDORA
from models.networks import CNN, MLPMnist
from utils.params import args_parser
from utils.helper import set_seed, set_logger, get_device
from evaluation import eval_model
from utils.load_rotate_mnist import BaseNodes_RotateMNIST


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = torch.sum(adj, dim=1)
    rowsum = torch.flatten(rowsum)
    d_inv_sqrt = torch.pow(rowsum, -1.0)
    d_inv_sqrt[d_inv_sqrt == float('inf')] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(d_mat_inv_sqrt, adj)


def run_FEDORA(args, device):
    if args.data_name == "mnist" or args.data_name == "fashion-mnist":
        nodes = BaseNodes_RotateMNIST(args.data_name, args.num_nodes, batch_size=args.batch_size)
        args.num_classes = 10

    args.num_nodes = len(nodes.train_loaders)
    if args.data_name == "mnist" or args.data_name == "fashion-mnist":
        g_net = MLPMnist(dim_in=28 * 28, out_dim=args.num_classes)
    else:
        raise ValueError("choose data_name from ['cifar10', 'cifar100']")

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
    criteria = torch.nn.CrossEntropyLoss()

    last_eval = -1
    best_step = -1
    best_acc = -1
    test_best_based_on_step, test_best_min_based_on_step = -1, -1
    test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
    step_iter = trange(args.num_steps)
    results = defaultdict(list)

    args.frac = 1.0
    similarity_matrix = torch.zeros(args.num_nodes, args.num_nodes)

    node_subspace = []
    node_y = []
    for i in range(args.num_nodes):
        data_loader = nodes.train_loaders[i]
        with torch.no_grad():
            x, y = [], []
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                x.append(data.view(data.shape[0], -1))
                y.append(target)
            x = torch.cat(x, dim=0)
            y = torch.cat(y, dim=-1)
        y_vector = torch.eye(args.num_classes, device=device) * 1
        x = torch.cat([x, y_vector[y]], dim=1)
        u_s, s_s, v_s = torch.svd_lowrank(x.t(), q=1)
        output_y = torch.unique(y, sorted=True)
        node_subspace.append(u_s)
        node_y.append(output_y)

    for i in range(args.num_nodes):
        for j in range(i+1, args.num_nodes):
            p_s, cospa, p_t = torch.svd(torch.mm(node_subspace[i].t(), node_subspace[j]))
            sim = torch.norm(cospa, 1)
            sim = torch.exp(sim*10)
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
    inv_similarity_matrix = (1-pa) * torch.linalg.inv(torch.eye(args.num_nodes) - pa * similarity_matrix)

    for step in step_iter:
        num_selected_nodes = max(int(args.frac * args.num_nodes), 1)
        if step == args.num_steps - 1:
            num_selected_nodes = args.num_nodes
        selected_node_ids = np.random.choice(range(args.num_nodes), num_selected_nodes, replace=False)

        w_aux = []
        lam_weights = torch.zeros(args.num_nodes)
        for node_id in selected_node_ids:
            client = LocalUpdateFEDORA(args=args, train_loader=nodes.train_loaders[node_id], val_loader=nodes.val_loaders[node_id])

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
                lam = torch.exp(lam/2)
            lam_weights[node_id] = lam

            w_local, _ = client.train(net=net_local.to(device), device=device, w_fedora=copy.deepcopy(w_globals[node_id]), lam=lam)

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
                    w_global_new[idx_i][k] += inv_similarity_matrix[idx_i, idx_j] * w_globals_old[idx_j][k]
        w_globals = copy.deepcopy(w_global_new)

        net_locals = []
        for node_id in range(args.num_nodes):
            net_local = copy.deepcopy(net_glob)
            net_local.load_state_dict(w_locals[node_id])
            net_locals.append(net_local)

        step_iter.set_description(f"Step: {step + 1}")
        if step % args.eval_every == 0:
            last_eval = step
            step_results, avg_loss, avg_acc, all_acc = eval_model(nodes, args.num_nodes, net_locals, criteria, device, split="test")
            logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}, {all_acc}")

            results['test_avg_loss'].append(avg_loss)
            results['test_avg_acc'].append(avg_acc)

            _, val_avg_loss, val_avg_acc, _ = eval_model(nodes, args.num_nodes, net_locals, criteria, device, split="val")
            if best_acc < val_avg_acc:
                best_acc = val_avg_acc
                best_step = step
                test_best_based_on_step = avg_acc
                test_best_min_based_on_step = np.min(all_acc)
                test_best_max_based_on_step = np.max(all_acc)
                test_best_std_based_on_step = np.std(all_acc)

            results['val_avg_loss'].append(val_avg_loss)
            results['val_avg_acc'].append(val_avg_acc)
            results['best_step'].append(best_step)
            results['best_val_acc'].append(best_acc)
            results['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
            results['test_best_min_based_on_step'].append(test_best_min_based_on_step)
            results['test_best_max_based_on_step'].append(test_best_max_based_on_step)
            results['test_best_std_based_on_step'].append(test_best_std_based_on_step)

    if step != last_eval:
        _, val_avg_loss, val_avg_acc, _ = eval_model(nodes, args.num_nodes, net_locals, criteria, device, split="val")
        step_results, avg_loss, avg_acc, all_acc = eval_model(nodes, args.num_nodes, net_locals, criteria, device, split="test")
        logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}, {all_acc}")

        results['test_avg_loss'].append(avg_loss)
        results['test_avg_acc'].append(avg_acc)

        if best_acc < val_avg_acc:
            best_acc = val_avg_acc
            best_step = step
            test_best_based_on_step = avg_acc
            test_best_min_based_on_step = np.min(all_acc)
            test_best_max_based_on_step = np.max(all_acc)
            test_best_std_based_on_step = np.std(all_acc)

        results['val_avg_loss'].append(val_avg_loss)
        results['val_avg_acc'].append(val_avg_acc)
        results['best_step'].append(best_step)
        results['best_val_acc'].append(best_acc)
        results['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
        results['test_best_min_based_on_step'].append(test_best_min_based_on_step)
        results['test_best_max_based_on_step'].append(test_best_max_based_on_step)
        results['test_best_std_based_on_step'].append(test_best_std_based_on_step)


if __name__ == '__main__':
    args = args_parser()
    set_logger()
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)
    run_FEDORA(args, device)

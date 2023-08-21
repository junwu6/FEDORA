import logging
import torch
import copy
import numpy as np
from tqdm import trange
from collections import defaultdict
from models.ClientUpdate import LocalUpdate
from models.networks import CNN, MLPMnist
from utils.params import args_parser
from utils.helper import set_seed, set_logger, get_device
from evaluation import eval_model
from utils.load_rotate_mnist import BaseNodes_RotateMNIST


def run_LOCAL(args, device):
    if args.data_name == "mnist" or args.data_name == "fashion-mnist":
        nodes = BaseNodes_RotateMNIST(args.data_name, args.num_nodes, batch_size=args.batch_size)

    args.num_nodes = len(nodes.train_loaders)
    if args.data_name == "mnist" or args.data_name == "fashion-mnist":
        g_net = MLPMnist()
    else:
        raise ValueError("choose data_name from ['cifar10', 'cifar100']")

    net_glob = g_net.to(device)
    net_glob.train()
    w_locals = {}
    for user in range(args.num_nodes):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict
    criteria = torch.nn.CrossEntropyLoss()

    last_eval = -1
    best_step = -1
    best_acc = -1
    test_best_based_on_step, test_best_min_based_on_step = -1, -1
    test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
    step_iter = trange(args.num_steps)
    results = defaultdict(list)

    args.frac = 1.0
    for step in step_iter:
        num_selected_nodes = max(int(args.frac * args.num_nodes), 1)
        if step == args.num_steps - 1:
            num_selected_nodes = args.num_nodes
        selected_node_ids = np.random.choice(range(args.num_nodes), num_selected_nodes, replace=False)
        for node_id in selected_node_ids:
            client = LocalUpdate(args=args, train_loader=nodes.train_loaders[node_id])
            net_local = copy.deepcopy(net_glob)
            w_local = copy.deepcopy(w_locals[node_id])
            net_local.load_state_dict(w_local)
            w_local, _ = client.train(net=net_local, device=device)

            for k, key in enumerate(net_glob.state_dict().keys()):
                w_locals[node_id][key] = copy.deepcopy(w_local[key])

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
    run_LOCAL(args, device)

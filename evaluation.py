from collections import defaultdict
import numpy as np
import torch.utils.data

from models.ClientUpdare import validate


def eval_model(nodes, num_nodes, nets, device, split, stacks=2):
    curr_results = evaluate(nodes, num_nodes, nets, device, split=split, stacks=stacks)
    avg_train_loss = np.mean([val["train_loss"] for val in curr_results.values()])
    avg_smape_loss = np.mean([val["smape_loss"] for val in curr_results.values()])
    avg_mae_loss = np.mean([val["mae_loss"] for val in curr_results.values()])
    avg_mse_loss = np.mean([val["mse_loss"] for val in curr_results.values()])
    avg_rmse_loss = np.mean([val["rmse_loss"] for val in curr_results.values()])
    avg_r2_loss = np.mean([val["r2_loss"] for val in curr_results.values()])

    return (
        curr_results,
        avg_train_loss,
        avg_smape_loss,
        avg_mae_loss,
        avg_mse_loss,
        avg_rmse_loss,
        avg_r2_loss,
    )


@torch.no_grad()
def evaluate(nodes, num_nodes, nets, device, split="test", stacks=2):
    results = defaultdict(lambda: defaultdict(list))
    for node_id in range(num_nodes):  # iterating over nodes
        if split == "test":
            curr_data = nodes.test_loaders[node_id]
        elif split == "val":
            curr_data = nodes.val_loaders[node_id]
        else:
            curr_data = nodes.train_loaders[node_id]

        net = nets[node_id]
        train_loss, smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = validate(
            net, curr_data, device, stacks
        )

        results[node_id]["train_loss"] = train_loss / len(curr_data.dataset)
        results[node_id]["smape_loss"] = smape_loss
        results[node_id]["mae_loss"] = mae_loss
        results[node_id]["mse_loss"] = mse_loss
        results[node_id]["rmse_loss"] = rmse_loss
        results[node_id]["r2_loss"] = r2_loss

    return results

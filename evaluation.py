from collections import defaultdict
import numpy as np
import torch.utils.data


def eval_model(nodes, num_nodes, nets, criteria, device, split):
    curr_results = evaluate(nodes, num_nodes, nets, criteria, device, split=split)
    avg_loss = np.mean([val['loss'] for val in curr_results.values()])

    all_acc = [val['correct'] / val['total'] for val in curr_results.values()]
    avg_acc = np.mean(all_acc)

    return curr_results, avg_loss, avg_acc, all_acc


@torch.no_grad()
def evaluate(nodes, num_nodes, nets, criteria, device, split='test'):
    results = defaultdict(lambda: defaultdict(list))
    for node_id in range(num_nodes):  # iterating over nodes
        running_loss, running_correct, running_samples = 0., 0., 0.
        if split == 'test':
            curr_data = nodes.test_loaders[node_id]
        elif split == 'val':
            curr_data = nodes.val_loaders[node_id]
        else:
            curr_data = nodes.train_loaders[node_id]

        for batch_count, batch in enumerate(curr_data):
            img, label = tuple(t.to(device) for t in batch)
            net = nets[node_id]
            net.eval()
            pred = net(img)
            running_loss += criteria(pred, label).item()
            running_correct += pred.argmax(1).eq(label).sum().item()
            running_samples += len(label)

        results[node_id]['loss'] = running_loss / (batch_count + 1)
        results[node_id]['correct'] = running_correct
        results[node_id]['total'] = running_samples

    return results

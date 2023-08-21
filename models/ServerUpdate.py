import copy


def ServerUpdate(w):
    num_batches = 0
    for i in range(len(w)):
        num_batches += w[i][1]
    w_avg = copy.deepcopy(w[0][0])

    for k in w_avg.keys():
        w_avg[k] = 0
        for i in range(len(w)):
            w_avg[k] += w[i][0][k] * w[i][1] / num_batches
    return w_avg



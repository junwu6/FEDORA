import os
import copy
import numpy as np
import torch
import torch.nn as nn

from models.forecasting.early_stop import EarlyStopping


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.0) / (len(x)))


def smooth_l1_loss(input, target, beta=1.0 / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 1:
        lr_adjust = {epoch: args.lr * (0.95 ** (epoch // 1))}

    elif args.lradj == 2:
        lr_adjust = {
            0: 0.0001,
            5: 0.0005,
            10: 0.001,
            20: 0.0001,
            30: 0.00005,
            40: 0.00001,
            70: 0.000001,
        }

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))
    else:
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
    return lr


def validate(net, dataloader, device, stacks=2):
    L1Loss = True
    criterion = smooth_l1_loss if L1Loss else nn.MSELoss(size_average=False).cuda()

    net.eval()

    losses_train = []
    losses_smape = []
    losses_mae = []
    losses_mse = []
    losses_rmse = []
    losses_r2 = []

    for _data in dataloader:
        inputs, targets = _data
        inputs = inputs.to(device)  # [batch_size, window_size, n_var]
        targets = targets.to(device)  # [batch_size, horizon, n_var]
        with torch.no_grad():
            if stacks == 1:
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            elif stacks == 2:
                outputs, mid = net(inputs)
                loss = criterion(outputs, targets) + criterion(mid, targets)
        # sMAPE
        absolute_percentage_errors = (
            2 * torch.abs(outputs - targets) / (torch.abs(outputs) + torch.abs(targets))
        )
        loss_smape = torch.mean(absolute_percentage_errors) * 100
        # MAE
        loss_mae = torch.mean(torch.abs(outputs - targets))
        # MSE
        loss_mse = torch.mean((outputs - targets) ** 2)
        # RMSE
        loss_rmse = torch.sqrt(loss_mse)
        # R squared
        loss_r2 = 1 - torch.sum((targets - outputs) ** 2) / torch.sum(
            (targets - torch.mean(targets)) ** 2
        )

        losses_train.append(loss.item())
        losses_smape.append(loss_smape.item())
        losses_mae.append(loss_mae.item())
        losses_mse.append(loss_mse.item())
        losses_rmse.append(loss_rmse.item())
        losses_r2.append(loss_r2.item())

    train_loss = np.array(losses_train).mean()
    smape_loss = np.array(losses_smape).mean()
    mae_loss = np.array(losses_mae).mean()
    mse_loss = np.array(losses_mse).mean()
    rmse_loss = np.array(losses_rmse).mean()
    r2_loss = np.array(losses_r2).mean()

    return train_loss, smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss


class FLocalUpdateFEDORA:
    def __init__(self, args, train_loader=None, val_loader=None, node_id=0):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.node_id = node_id

        self.epochs = args.inner_steps
        self.patience = args.patience
        self.verbose = args.verbose
        self.eval_every = args.eval_every
        self.stacks = args.num_stacks
        self.input_dim = args.input_size
        self.L1Loss = args.L1Loss
        self.decompose = args.decompose

        weights_folder = "weights"
        os.makedirs(weights_folder, exist_ok=True)
        self.checkpoint_path = os.path.join(weights_folder, f"checkpoint_{node_id}.pt")

        self.early_stopping = EarlyStopping(
            patience=self.patience,
            verbose=False,
            checkpoint_path=self.checkpoint_path,
        )
        self.criterion = (
            smooth_l1_loss if self.L1Loss else nn.MSELoss(size_average=False).cuda()
        )
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

    def train(self, net, device, w_fedora=None, lam=1):

        optim = torch.optim.Adam(
            params=net.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-5,
        )
        epoch_start = 0
        loss_evol = []

        for epoch in range(epoch_start, self.epochs):
            net.train()
            epoch_loss = 0.0
            self.args.lradj = 2
            adjust_learning_rate(optim, epoch, args=self.args)

            for data in self.train_loader:
                inputs, targets = data
                w_0 = copy.deepcopy(net.state_dict())
                inputs = inputs.to(device)  # [batch_size, seq_len, input_size=n_var]
                targets = targets.to(device)  # [batch_size, horizon, input_size]

                # inference
                net.zero_grad()
                if self.stacks == 1:
                    forecast = net(inputs)
                    loss = self.criterion(forecast, targets)
                if self.stacks == 2:
                    forecast, mid = net(inputs)
                    loss = self.criterion(forecast, targets) + self.criterion(
                        mid, targets
                    )
                epoch_loss += loss.item()

                # backpropagate
                loss.backward()
                optim.step()

                # FEDORA custom
                if w_fedora is not None:
                    w_net = copy.deepcopy(net.state_dict())
                    for key in w_net.keys():
                        w_net[key] = w_net[key] - self.args.inner_lr * lam * (
                            w_0[key] - w_fedora[key]
                        )
                        net.load_state_dict(w_net)
                        optim.zero_grad()

            epoch_loss /= len(self.train_loader)  # average loss per batch
            loss_evol.append(epoch_loss)

            # valid
            _, smape_loss, mae_loss, mse_loss, rmse_loss, r2_loss = validate(
                net, self.val_loader, device, stacks=self.stacks
            )

            if self.verbose:
                if epoch % self.eval_every == 0:
                    print(f"epoch: {epoch}, Train loss: {epoch_loss:.7f}")
                    print(
                        f"Eval: smape={smape_loss:.7f}, mae={mae_loss:.7f}, mse={mse_loss:.7f}, rmse={rmse_loss:.7f}, r2={r2_loss:.7f}"
                    )

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            self.early_stopping(mse_loss, net)

            if self.early_stopping.early_stop:
                break

        # load the last checkpoint with the best model (saved by EarlyStopping)
        net.load_state_dict(torch.load(self.checkpoint_path))

        return net.state_dict(), len(self.train_loader)

    def get_loss(self, net, device, stacks=2):
        net.eval()
        loss = 0.0
        for data in self.train_loader:
            inputs, targets = data
            inputs = inputs.to(device)  # [batch_size, seq_len, input_size=n_var]
            targets = targets.to(device)  # [batch_size, horizon, input_size]

            if stacks == 1:
                forecast = net(inputs)
                loss = self.criterion(forecast, targets)
            if stacks == 2:
                forecast, mid = net(inputs)
                loss = self.criterion(forecast, targets) + self.criterion(mid, targets)
            loss += loss.item()
        return loss

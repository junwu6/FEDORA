import torch
from torch import nn
import copy


class LocalUpdate(object):
    def __init__(self, args, train_loader=None):
        self.args = args
        self.train_loader = train_loader
        self.criteria = nn.CrossEntropyLoss()

    def train(self, net, device):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.inner_lr, momentum=.9, weight_decay=self.args.inner_wd)

        for i in range(self.args.inner_steps):
            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)

                net.zero_grad()
                log_probs = net(images)
                loss = self.criteria(log_probs, labels)
                loss.backward()
                optimizer.step()

        return net.state_dict(), len(self.train_loader)

    def get_loss(self, net, device):
        net.eval()
        loss = 0.
        for images, labels in self.train_loader:
            images, labels = images.to(device), labels.to(device)
            log_probs = net(images)
            loss += self.criteria(log_probs, labels)
        return loss


class LocalUpdateFEDORA(object):
    def __init__(self, args, train_loader=None, val_loader=None):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criteria = nn.CrossEntropyLoss()

    def train(self, net, device, w_fedora=None, lam=1):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.inner_lr, momentum=.9, weight_decay=self.args.inner_wd)

        for i in range(self.args.inner_steps):
            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)
                w_0 = copy.deepcopy(net.state_dict())

                log_probs = net(images)
                loss = self.criteria(log_probs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if w_fedora is not None:
                    w_net = copy.deepcopy(net.state_dict())
                    for key in w_net.keys():
                        w_net[key] = w_net[key] - self.args.inner_lr * lam * (w_0[key] - w_fedora[key])
                    net.load_state_dict(w_net)
                    optimizer.zero_grad()

        return net.state_dict(), len(self.train_loader)

    def get_loss(self, net, device):
        net.eval()
        loss = 0.
        for images, labels in self.val_loader:
            images, labels = images.to(device), labels.to(device)
            log_probs = net(images)
            loss += self.criteria(log_probs, labels)
        return loss

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import os

args = {
    'batch_size': 32,
    'test_batch_size': 1000,
    'epochs': 10,
    'lr': 0.01,
    'momentum': 0.9,
    'seed': 1,
    'log_interval': 10
}
use_cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if use_cuda else 'cpu')


class SimulationNet(nn.Module):
    def __init__(self):
        super(SimulationNet, self).__init__()
        self.L1 = nn.Linear(20, 64)
        self.L2 = nn.Linear(64, 200)
        self.L3 = nn.Linear(200, 104)

    def forward(self, x):
        x1 = F.relu(self.L1(x))
        x2 = F.relu(self.L2(x1))
        x3 = F.relu(self.L3(x2))
        return x3


def train_epoch(epoch, model, loader, optimizer):
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(loader.dataset),
                            100. * batch_idx / len(loader), loss.item()))


def train(model, loader):
    torch.manual_seed(args['seed'])
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
    for epoch in range(1, args['epochs']+1):
        train_epoch(args['epochs'], model, loader, optimizer)


if __name__ == '__main__':
    net = SimulationNet().to(device)
    # 输入训练集
    train_data = Data.TensorDataset(torch.Tensor(), torch.Tensor())
    loader = Data.DataLoader(
        dataset=train_data,
        batch_size=args['batch_size'],
        shuffle=True
    )
    train(net, loader)







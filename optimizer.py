import torch
from torch import nn
from torch.optim import Optimizer
from itertools import chain
from torch.optim import Adam
from collections import defaultdict


class Lookahead(Optimizer):
    """pytorch implementation of 'k step forward, 1 step back'
    args:
        base_optimizer: torch.optim, the base optimizer
        k: int, step of forward
        alpha: float, ratio of interpolation
    """
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.optimizer = base_optimizer
        self.k = k
        self.alpha = alpha

        self.param_groups = base_optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = base_optimizer.state

        for group in self.param_groups:
            group['counter'] = 0
    
    def update(self, group):
        for fast in group['params']:
            param_state = self.state[fast]
            if 'slow_param' not in param_state:
                param_state['slow_param'] = torch.zeros_like(fast.data)
                param_state['slow_param'].copy_(fast.data)
            slow = param_state['slow_param']
            slow += self.alpha * (fast.data - slow)
            fast.data.copy_(slow)
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group['counter'] == 0:
                self.update(group)
            group['counter'] += 1
            if group['counter'] == self.k:
                group['counter'] = 0
        return loss





'''class TryModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super(TryModel,self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        output = self.fc1(x)
        output = self.fc2(output)
        return output

torch.manual_seed(1)

x = torch.randn(64,100)
y = torch.randn(64,5)
model = TryModel(100,5)
criterion = nn.MSELoss()
opt = Lookahead(Adam(model.parameters()))
losses_ahead = []
for i in range(2000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    losses_ahead.append(float(loss.data))
    opt.zero_grad()
    loss.backward()
    opt.step()


model = TryModel(100,5)
criterion = nn.MSELoss()
opt = Adam(model.parameters())
losses_adam = []
for i in range(2000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    losses_adam.append(float(loss.data))
    opt.zero_grad()
    loss.backward()
    opt.step()'''




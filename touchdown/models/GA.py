import torch
from torch import nn


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 128, kernel_size=8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 64, kernel_size=4, stride=2),
                                  nn.ReLU())
        self.fcl = nn.Linear(64 * 11 * 11, 64)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64 * 11 * 11)
        return self.fcl(x)


class GA(nn.Module):
    def __init__(self, opts):
        super(GA, self).__init__()
        self.opts = opts
        self.Conv = Conv()
        self.instr_map = nn.Linear(256, 64)
        self.fcl = nn.Sequential(nn.Linear(64, 256),
                                 nn.ReLU())
        self.time_embed = nn.Embedding(self.opts.max_route_len, 32)
        self.rnn = nn.LSTM(256, 256, 1, batch_first=True)
        self.policy = nn.Linear(256 + 32, 4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, I, h_t, c_t, t):
        I = self.Conv(I)
        g = torch.sigmoid(self.instr_map(x)).squeeze(1)
        u = torch.mul(I, g)
        v = self.fcl(u).unsqueeze(1)
        _, (h_t, c_t) = self.rnn(v, (h_t, c_t))
        t = self.time_embed(t)
        t_expand = torch.zeros(x.size(0), 32).to(self.device)
        t_expand.copy_(t)
        policy_input = torch.cat([h_t.squeeze(0), t_expand], dim=1)
        a_t = self.policy(policy_input)
        return a_t, (h_t, c_t)

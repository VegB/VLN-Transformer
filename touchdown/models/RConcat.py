import torch
from torch import nn
from models.rnn import CustomRNN
from utils import padding_idx


class Conv_net(nn.Module):
    def __init__(self, opts):
        super(Conv_net, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opts = opts
        self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=4),
                nn.ReLU())
        self.fcl = nn.Linear(6 * 6 * 64, self.opts.hidden_dim)
        
    def forward(self, x):
        """
        :param x: [batch_size, 1, 100, 100]
        """
        x = self.conv(x)  # [batch_size, 64, 6, 6]
        x = x.view(-1, 6 * 6 * 64)  # [batch_size, 6 * 6 * 64]
        return self.fcl(x)  # [batch_size, 256]


class Embed_RNN(nn.Module):
    def __init__(self, vocab_size):
        super(Embed_RNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(vocab_size, 32, padding_idx)
        self.rnn_kwargs = {'cell_class': nn.LSTMCell,
                           'input_size': 32,
                           'hidden_size': 256,
                           'num_layers': 1,
                           'batch_first': True,
                           'dropout': 0,
                           }
        self.rnn = CustomRNN(**self.rnn_kwargs)
                
    def create_mask(self, batchsize, max_length, length):
        """Given the length create a mask given a padded tensor"""
        tensor_mask = torch.zeros(batchsize, max_length)
        for idx, row in enumerate(tensor_mask):
            row[:length[idx]] = 1
        return tensor_mask.to(self.device)
        
    def forward(self, x, lengths):
        x = self.embedding(x)  # [batchsize, max_length, 32]
        embeds_mask = self.create_mask(x.size(0), x.size(1), lengths)
        x, _ = self.rnn(x, mask=embeds_mask)  # [batch_size, max_length, 256]
        x = x[:, -1, :]  # [batch_size, 256]
        return x.unsqueeze(1)  # [batch_size, 1, 256]

   
class RConcat(nn.Module):
    def __init__(self, opts):
        super(RConcat, self).__init__()
        self.opts = opts
        self.Conv = Conv_net(self.opts)
        self.action_embed = nn.Embedding(4, 16)
        self.rnn = nn.LSTM(256 + 256 + 16, 256, 1, batch_first=True)
        self.time_embed = nn.Embedding(self.opts.max_route_len, 32)
        self.policy = nn.Linear(256 + 32, 4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x, I, a, h_t, c_t, t):
        """
        :param x: [batch_size, 1, 256], encoded instruction
        :param I: [batch_size, 1, 100, 100], features
        :param a: [batch_size, 1], action
        :param h_t: [1, batch_size, 256], hidden state in LSTM
        :param c_t: [1, batch_size, 256], memory in LSTM
        :param t:
        :return:
        """
        I = self.Conv(I).unsqueeze(1)  # [batch_size, 1, 256]
        a = self.action_embed(a)  # [batch_size, 1, 16]
        s_t = torch.cat([x, I, a], dim=2)  # [batch_size, 1, 256 + 256 + 16]
        _, (h_t, c_t) = self.rnn(s_t, (h_t, c_t))
        t = self.time_embed(t)
        t_expand = torch.zeros(x.size(0), 32).to(self.device)
        t_expand.copy_(t)
        policy_input = torch.cat([h_t.squeeze(0), t_expand], dim=1)  # [batch_size, 256 + 32]
        a_t = self.policy(policy_input)
        return a_t, (h_t, c_t)

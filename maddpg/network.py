import os
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init


class CriticNetwork(nn.Module):

    def __init__(self, lr, state_dim, action_dims, fc1_dim, fc2_dim, name,
                 chkpt_dir):
        super().__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        os.makedirs(chkpt_dir, exist_ok=True)

        self.fc1 = nn.Linear(state_dim + sum(action_dims), fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

        # TODO: 正交初始化g权重
        # self.q = nn.Sequential(
        #     nn.Linear(state_dim, sum(action_dims), fc1_dim),
        #     nn.ReLU()
        #     nn.Linear(fc1_dim, fc2_dim),
        #     nn.ReLU()
        # )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, joint_action):
        x = F.relu(self.fc1(th.cat([state, joint_action], dim=1)))
        x = F.relu(self.fc2(x))
        return self.q(x)

    def save_checkpoint(self):
        th.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(th.load(self.chkpt_file, weights_only=True))


class ActorNetwork(nn.Module):

    def __init__(self, lr, obs_dim, action_dim, fc1_dim, fc2_dim, action_low,
                 action_high, name, chkpt_dir):
        super().__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        os.makedirs(chkpt_dir, exist_ok=True)

        self.fc1 = nn.Linear(obs_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.pi = nn.Linear(fc2_dim, action_dim)

        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.action_low = th.tensor(action_low,
                                    dtype=th.float,
                                    device=self.device)
        self.action_high = th.tensor(action_high,
                                     dtype=th.float,
                                     device=self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        coarse_action = F.tanh(self.pi(x))
        action = self.action_low + (
            (self.action_high - self.action_low) / 2) * (coarse_action + 1)

        # print(action)
        # assert all((action >= self.action_low) & (action <= self.action_high)), f'action boundary wrong, {self.action_low} - {self.action_high}'
        return action

    def save_checkpoint(self):
        th.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(th.load(self.chkpt_file, weights_only=True))

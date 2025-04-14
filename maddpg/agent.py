import torch as th
from network import CriticNetwork, ActorNetwork
import copy
import numpy as np


class Agent:
    def __init__(self, state_dim, action_dims, obs_dim, action_dim, agent_name, chkpt_dir,\
                 action_low, action_high,\
                 lr_a=1e-2, lr_c=1e-2, fc1_dim=64, fc2_dim=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.action_dims = action_dims
        self.agent_name = agent_name

        # self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        self.action_low = th.tensor(action_low, dtype=th.float, device='cpu')
        self.action_high = th.tensor(action_high, dtype=th.float, device='cpu')

        self.actor = ActorNetwork(lr_a, obs_dim, action_dim, fc1_dim, fc2_dim, action_low, \
                                  action_high, agent_name + '_actor', chkpt_dir)
        self.critic = CriticNetwork(lr_c, state_dim, action_dims, fc1_dim, fc2_dim, \
                                    agent_name + '_critic', chkpt_dir)

        self.target_actor = ActorNetwork(lr_a, obs_dim, action_dim, fc1_dim, fc2_dim, action_low, \
                                  action_high, agent_name + '_target_actor', chkpt_dir)
        self.target_critic = CriticNetwork(lr_c, state_dim, action_dims, fc1_dim, fc2_dim, \
                                    agent_name + '_target_critic', chkpt_dir)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_actor.eval()
        self.target_critic.eval()

    def choose_action(self, obs):
        obs = th.tensor(np.array([obs]), dtype=th.float).to(self.actor.device)
        action = self.actor(obs)
        action = th.squeeze(action).detach().cpu()

        # noise = th.randn_like(action)
        # action = action + noise

        # # 加噪音后需要再修剪 action
        # action = th.clamp(action, self.action_low, self.action_high)

        return action.numpy()

    def update_parameters(self):
        self.polyak_update(self.actor.parameters(),
                           self.target_actor.parameters())
        self.polyak_update(self.critic.parameters(),
                           self.target_critic.parameters())

    def polyak_update(self, params, target_params):
        with th.no_grad():
            for param, target_param in zip(params, target_params):
                target_param.data.mul_(1 - self.tau)
                th.add(target_param.data,
                       param.data,
                       alpha=self.tau,
                       out=target_param.data)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

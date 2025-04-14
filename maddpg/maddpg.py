import numpy as np
import torch as th
import torch.nn.functional as F
from agent import Agent
from buffer import MultiAgentReplayBuffer
from utils import plot_reward_curve
from tqdm.notebook import tqdm
from torch.nn.utils import clip_grad_norm_


class MADDPG:

    def __init__(self,
                 env,
                 lr_a=0.01,
                 lr_c=0.01,
                 fc1_dim=64,
                 fc2_dim=64,
                 gamma=0.99,
                 tau=0.01,
                 chkpt_dir='tmp/maddpg/',
                 buffer_size=1000000,
                 batch_size=1024,
                 grad_clip=0.5):
        self.env = env
        self.agent_names = env.agents
        self.n_agents = env.num_agents
        self.state_dim = np.prod(env.state().shape)
        self.action_dims = [
            np.prod(env.action_space(a).shape) for a in self.agent_names
        ]
        self.obs_dims = [
            np.prod(env.observation_space(a).shape) for a in self.agent_names
        ]
        self.action_boundary = [(env.action_space(a).low,
                                 env.action_space(a).high) for a in env.agents]
        self.gamma = gamma

        self.agents = []
        for i in range(self.n_agents):
            self.agents.append(Agent(self.state_dim, self.action_dims, self.obs_dims[i], self.action_dims[i], \
                                     self.agent_names[i], chkpt_dir, self.action_boundary[i][0], self.action_boundary[i][1], \
                                        lr_a, lr_c, fc1_dim, fc2_dim, gamma, tau))

        self.replay_buffer = MultiAgentReplayBuffer(buffer_size,
                                                    self.state_dim,
                                                    self.obs_dims,
                                                    self.action_dims,
                                                    self.n_agents, batch_size)

        self.grad_norm_clip = grad_clip

    def choose_action(self, raw_obs, noise_scale):
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[i])

            noise = np.random.randn(*action.shape).astype(
                action.dtype) * noise_scale
            action += noise
            action = np.clip(action, self.action_boundary[i][0],
                             self.action_boundary[i][1])

            actions.append(action)

        return actions

    def learn(self, memory):
        if not memory.ready():
            # print('Not ready!')
            return

        state, obs, actions, rewards, dones, state_, obs_ = memory.sample()
        device = self.agents[0].actor.device

        state = th.tensor(state, dtype=th.float).to(device)
        # obs = th.tensor(obs, dtype=th.float).to(device)
        # action = th.tensor(action, dtype=th.float).to(device)
        # reward = th.tensor(reward, dtype=th.float).to(device)
        state_ = th.tensor(state_, dtype=th.float).to(device)
        # obs_ = th.tensor(obs_, dtype=th.float).to(device)
        # done = th.tensor(done, dtype=th.float).to(device)

        action = []
        # mu_action = []
        mu_action_ = []
        for i in range(self.n_agents):
            action.append(th.tensor(actions[i], dtype=th.float).to(device))

            # # 当前时间步的动作预测使用主网络
            # old_obs = th.tensor(obs[i], dtype=th.float).to(device)
            # mu_action.append(self.agents[i].actor(old_obs))

            # 下一时间步的动作预测使用目标网络
            new_obs = th.tensor(obs_[i], dtype=th.float).to(device)
            mu_action_.append(self.agents[i].target_actor(new_obs))

        joint_action = th.cat(action, dim=1)
        # joint_mu_action = th.cat(mu_action, dim=1)
        joint_mu_action_ = th.cat(mu_action_, dim=1)
        for i in range(self.n_agents):
            reward = th.tensor(rewards[i], dtype=th.float).to(device).flatten()
            done = th.tensor(dones[i], dtype=th.int8).to(device).flatten()

            next_q_value = self.agents[i].target_critic(
                state_, joint_mu_action_).flatten()
            y_target = reward + (1 - done) * self.gamma * next_q_value
            y = self.agents[i].critic(state, joint_action).flatten()

            # TODO: 是否保存计算图？
            critic_loss = F.mse_loss(y, y_target)
            self.agents[i].critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            clip_grad_norm_(self.agents[i].critic.parameters(),
                            self.grad_norm_clip)

            # for name, param in self.agents[i].critic.named_parameters():
            #     if param.grad is not None:
            #         print(
            #             f"{name} gradient mean: {param.grad.mean().item()}, max: {param.grad.max().item()}"
            #         )
            #     else:
            #         print(
            #             f"{name} has no gradient (unused or not backpropagated)"
            #         )

            self.agents[i].critic.optimizer.step()

            # 当前时间步的动作预测使用主网络
            old_obs = th.tensor(obs[i], dtype=th.float).to(device)
            mu_action = self.agents[i].actor(old_obs)
            mixture_action = action.copy()
            mixture_action[i] = mu_action

            q_value = self.agents[i].critic(state, th.cat(mixture_action,
                                                          dim=1)).flatten()
            actor_loss = -q_value.mean()
            self.agents[i].actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            clip_grad_norm_(self.agents[i].actor.parameters(),
                            self.grad_norm_clip)

            # for name, param in self.agents[i].actor.named_parameters():
            #     if param.grad is not None:
            #         print(
            #             f"{name} gradient mean: {param.grad.mean().item()}, max: {param.grad.max().item()}"
            #         )
            #     else:
            #         print(
            #             f"{name} has no gradient (unused or not backpropagated)"
            #         )

            self.agents[i].actor.optimizer.step()

        for i in range(self.n_agents):
            self.agents[i].update_parameters()

    def train(self,
              n_episodes,
              train_freq=100,
              fig_name='img/simple_adversary.svg',
              seed=42):
        env = self.env

        for a in self.agent_names:
            # print(f'\033[92m\t{a}\033[0m', end='')
            print(f'\t{a}', end='')
        print()

        n_steps = 0
        reward_data = {k: [] for k in self.agent_names}
        for episode in tqdm(range(n_episodes)):
            obs, _ = env.reset(seed)
            state = env.state()
            data = dict.fromkeys(self.agent_names, 0)
            noise_scale = 1 - episode / n_episodes

            while True:
                # TODO: 噪音应该越来越小
                action = self.choose_action(list(obs.values()), noise_scale)
                obs_, reward, termination, truncation, _ = env.step(
                    dict(zip(self.agent_names, action)))
                new_state = env.state()

                # print(action)

                termination = np.array(list(termination.values()))
                truncation = np.array(list(truncation.values()))
                done = termination | truncation
                self.replay_buffer.store_transition(state, list(obs.values()),
                                                    action,
                                                    list(reward.values()),
                                                    new_state,
                                                    list(obs_.values()), done)

                state = new_state
                obs = obs_
                for key, val in zip(data.keys(), reward.values()):
                    data[key] += val

                n_steps += 1
                # print(n_steps)
                if n_steps % train_freq == 0:
                    self.learn(self.replay_buffer)

                if any(done):
                    break
            for rew_ls, val in zip(reward_data.values(), data.values()):
                rew_ls.append(val)

            if (episode + 1) % 50 == 0:
                for a in self.agent_names:
                    # print(
                    #     f'\033[92m\t{sum(reward_data[a][-50:]) / 50:.2f}\033[0m',
                    #     end='')
                    print(f'\t{sum(reward_data[a][-50:]) / 50:.2f}', end='')
                print(f'\t[{episode + 1}/{n_episodes}]')

        plot_reward_curve(reward_data, fig_name)

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

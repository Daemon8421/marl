import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, max_size, state_dim, obs_dims, action_dims, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.N = n_agents
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.batch_size = batch_size

        self.init_agent_memory()
        self.state_mem = np.zeros((self.mem_size, state_dim))
        self.new_state_mem = np.zeros((self.mem_size, state_dim))

    def init_agent_memory(self):
        self.agent_obs_mem = []
        self.agent_new_obs_mem = []
        self.agent_action_mem = []
        self.agent_reward_mem = []
        self.agent_done_mem = []

        for i in range(self.N):
            # 使用 for 循环而不是直接增加维度的原因：每个智能体的obs、action维度可能不同
            self.agent_obs_mem.append(np.zeros((self.mem_size, self.obs_dims[i])))
            self.agent_new_obs_mem.append(np.zeros((self.mem_size, self.obs_dims[i])))
            self.agent_action_mem.append(np.zeros((self.mem_size, self.action_dims[i])))
            
            self.agent_reward_mem.append(np.zeros((self.mem_size, 1)))
            self.agent_done_mem.append(np.zeros((self.mem_size, 1), dtype=np.int8))
    
    def store_transition(self, state, raw_obs, actions, rewards, state_, raw_obs_, dones):
        idx = self.mem_cntr % self.mem_size

        self.state_mem[idx] = state
        self.new_state_mem[idx] = state_

        # print(type(actions), type(actions[0]), type(self.agent_action_mem[0][0]))

        for i in range(self.N):
            self.agent_obs_mem[i][idx] = raw_obs[i]
            self.agent_new_obs_mem[i][idx] = raw_obs_[i]
            self.agent_action_mem[i][idx] = actions[i]
            self.agent_reward_mem[i][idx] = rewards[i]
            self.agent_done_mem[i][idx] = dones[i]
        
        self.mem_cntr += 1
    
    def sample(self):
        real_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.permutation(real_mem)[:self.batch_size]

        state = self.state_mem[batch]
        state_ = self.new_state_mem[batch]

        agent_obs = []
        agent_obs_ = []
        agent_action = []
        agent_reward = []
        agent_done = []
        for i in range(self.N):
            agent_obs.append(self.agent_obs_mem[i][batch])
            agent_obs_.append(self.agent_new_obs_mem[i][batch])
            agent_action.append(self.agent_action_mem[i][batch])
            agent_reward.append(self.agent_reward_mem[i][batch])
            agent_done.append(self.agent_done_mem[i][batch])

        return state, agent_obs, agent_action, agent_reward, agent_done, state_, agent_obs_

    def ready(self):
        return self.mem_cntr >= self.batch_size
from pettingzoo.mpe import simple_tag_v3, simple_adversary_v3
from maddpg import MADDPG

if __name__ == '__main__':
    seed = 42
    env = simple_adversary_v3.parallel_env(render_mode=None, continuous_actions=True)
    env.reset(seed)
    model = MADDPG(env)
    model.train(10000)
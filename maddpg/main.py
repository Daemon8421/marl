from pettingzoo.mpe import simple_tag_v3, simple_adversary_v3
from maddpg import MADDPG
from utils import set_seed, gen_fig_name

if __name__ == '__main__':
    seed = 42
    n_episodes = 1000

    # set_seed(seed)
    fn = gen_fig_name(n_episodes)

    env = simple_adversary_v3.parallel_env(render_mode=None,
                                           continuous_actions=True)
    env.reset(seed)
    model = MADDPG(env)
    model.train(n_episodes)
    # model.save_checkpoint()

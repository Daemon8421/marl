import matplotlib.pyplot as plt
import numpy as np

def plot_reward_curve(reward_data, fig_name='img/simple_adversary.svg'):
    # 创建画布和坐标轴（设置DPI和尺寸使图像更清晰）
    plt.figure(figsize=(8, 5), dpi=100)

    # 设置颜色和线型（使用seaborn风格更美观）
    plt.style.use('seaborn-v0_8-whitegrid')

    n_episodes = len(next(iter(reward_data.values())))
    episodes = np.arange(1, n_episodes + 1)
    # 绘制每条折线
    for agent, rewards in reward_data.items():
        plt.plot(episodes, rewards, 
                label=agent,
                linewidth=2,
                alpha=0.8)

    # 添加标题和标签（使用更专业的字体设置）
    plt.title('Reward Trends by Agent', fontsize=14, pad=20)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)

    # 设置坐标轴范围（自动扩展5%的空白区域）
    plt.xlim(0.9, n_episodes + 0.1)
    plt.ylim(min(min(rewards) for rewards in reward_data.values()) * 1.05, max(max(rewards) for rewards in reward_data.values()) * 1.05)

    # 添加网格和图例（位置优化）
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', framealpha=1)

    # 调整布局并显示
    plt.tight_layout()
    plt.savefig(fig_name)
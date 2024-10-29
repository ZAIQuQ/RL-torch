import random
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import utils 
from SAC_sampling.sac import SACContinuous as SAC
                             

def main():
    env_name = 'Pendulum-v1' # 倒立摆任务环境v0已经过时，改为v1试试
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值
    print(f"[state_dim]: {state_dim}\n[action_dim]: {action_dim}\n[action_bound]: {action_bound}")

    random.seed(0)
    np.random.seed(0)
    # env.seed(0) 
    env.reset(seed=0) # 最新版本的 gym 中，seed()方法已被移除
    torch.manual_seed(0)

    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.99
    tau = 0.005  # 软更新参数
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64
    target_entropy = -env.action_space.shape[0]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    replay_buffer = utils.ReplayBuffer(buffer_size)
    agent = SAC(state_dim, hidden_dim, action_dim, action_bound,
                        actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                        gamma, device)

    return_list = utils.train_off_policy_agent(env, agent, num_episodes,
                                                replay_buffer, minimal_size,
                                                batch_size)
    
    # 绘图分析
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))
    plt.show()

    mv_return = utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))
    plt.show()

if __name__ == "__main__":
    main()
'''
Code inspired and readapted from https://github.com/jmichaux/dqn-pytorch
'''

import gym
import torch
import random
import math
import datetime
import os
import wandb
import logging as log
import numpy as np
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from atari_wrappers import make_atari, wrap_deepmind
from replay_buffer import ReplayBuffer
from dqn import DQN


def select_action(state, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)


def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)


def learn():
    obses_t, actions, rewards, obses_tp1, dones = buffer.sample(BATCH_SIZE)
    
    actions = [a.to(device) for a in actions]
    rewards = [r.to(device) for r in rewards]

    non_final_mask = torch.tensor([True if o is not None else False for o in obses_tp1], device=device)

    non_final_next_states = torch.cat([o for o in obses_tp1 if o is not None]).to(device)

    state_batch = torch.cat(obses_t).to(device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def update_target(model, target):
    target.load_state_dict(model.state_dict())


def train(render=False):
    episode_rewards = [0.0]

    obs = env.reset()
    state = get_state(obs)

    train_start = datetime.datetime.now()
    episode_start = train_start

    for steps_done in range(STEPS):

        action = select_action(state, steps_done)
        
        # take one action and observe the outcome
        obs, reward, done, info = env.step(action)
        
        episode_rewards[-1] += reward

        if not done:
            next_state = get_state(obs)
        else:
            next_state = None

        reward = torch.tensor([reward], device=device, dtype=torch.float32)

        # push to memory
        buffer.add(state, action.to('cpu'), reward.to('cpu'), next_state, done)

        # update current state
        state = next_state

        if done:
            episode_end = datetime.datetime.now()
            episode_time = episode_end - episode_start
            tot_time = episode_end - train_start
            episode_start = episode_end

            avg_50 = np.round(np.mean(episode_rewards[-50:]), 1)

            print(f'Total steps: {steps_done} \t Episodes: {len(episode_rewards)} \t Time: {tot_time} \t Episode Time: {episode_time} \t Reward: {episode_rewards[-1]} \t Reward Avg (Last 50): {avg_50}')
            log.info(f'Total steps: {steps_done} \t Episodes: {len(episode_rewards)} \t Time: {tot_time} \t Episode Time: {episode_time} \t Reward: {episode_rewards[-1]} \t Reward Avg (Last 50): {avg_50}')
            wandb.log({'reward': episode_rewards[-1], 'reward avg (last 50)': avg_50})

            obs = env.reset()
            state = get_state(obs)
            episode_rewards.append(0.0)
        
        if steps_done > INITIAL_MEMORY:
            if steps_done % LEARNING_FREQ == 0:
                learn()

            if steps_done % TARGET_UPDATE == 0:
                update_target(policy_net, target_net)

        if steps_done % SAVE_FREQ == 0:
            time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            torch.save(policy_net.state_dict(), models_dir + time)

    # close the environment
    env.close()


if __name__ == "__main__":

    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.01
    EPS_DECAY = 10000
    TARGET_UPDATE = 1000
    LEARNING_RATE = 1e-4
    LEARNING_FREQ = 4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY
    STEPS = int(1e7)
    SAVE_FREQ = 10000

    models_dir = '../../models/pong/'
    log_dir = '../../logs/pong/'

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb.init(project='dqn_pong')
    wandb.run.name = time

    log.basicConfig(filename=log_dir+time+'.log', level=log.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    log.info('Start training ...')

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    log.info(f'Using device: {device}')

    # define the environment
    env = make_atari('PongNoFrameskip-v4')
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    num_actions = env.action_space.n

    # agent and target
    policy_net = DQN(in_channels=4, num_actions=num_actions).to(device)
    target_net = DQN(in_channels=4, num_actions=num_actions).to(device)
    update_target(policy_net, target_net)

    # if there is a saved model
    if len(os.listdir(models_dir)) != 0:
        print(f'Loading a model: {os.listdir(models_dir)[-1]}')
        log.info(f'Loading a model: {os.listdir(models_dir)[-1]}')
        policy_net.load_state_dict(torch.load(models_dir + os.listdir(models_dir)[-1]))
        policy_net.eval()
        update_target(policy_net, target_net)

    # optimizer
    optimizer = optim.RMSprop(policy_net.parameters(), lr=LEARNING_RATE)

    # instantiate replay buffer
    buffer = ReplayBuffer(MEMORY_SIZE)

    # train
    train()
    # save trained model
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f'Saving a model: {os.listdir(models_dir)[-1]}')
    log.info(f'Saving a model: {os.listdir(models_dir)[-1]}')
    torch.save(policy_net.state_dict(), models_dir + time)

    wandb.run.save()
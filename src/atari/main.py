import gym
import torch
import random
import math
import datetime
import os
import wandb
import argparse
import time as t
import logging as log
import numpy as np
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

from gym.wrappers import Monitor
from atari_wrappers import make_atari, wrap_deepmind
from replay_buffer import ReplayBuffer
from dqn import DQN

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, '../deepfool')
from deepfool_atari import deepfool

sys.path.insert(0, '../simba')
from simba import simba, simba_mod

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')  # False by default, if --train then True
    parser.add_argument('--test', action='store_true')  # False by default, if --test then True
    parser.add_argument('--render', action='store_true')  # False by default, if --render then True
    parser.add_argument('--attack', help='noise/deepfool/simba/simba_mod')
    parser.add_argument('--game', help='game to train/test on')
    parser.add_argument('--video', action='store_true')  # False by default, if --train then True
    parser.add_argument('--imshow', action='store_true')  # False by default, if --train then True

    return parser.parse_args()


def select_action(state, steps_done):
    '''
    Select an action given a state using decaying epsilon greedy
    '''
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)
    wandb.log({'epsilon': eps_threshold})
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


def rand_noise(state, epsilon=0.02):
    '''
    Add some random noise to the image
    '''
    # get the noise to be in [-eps, eps]
    noise = -2 * epsilon * torch.rand(size=state.size()) + epsilon
    state += noise
    state /= torch.max(state)
    # plt.imshow(state.reshape(1, -1, 84).permute(1, 2, 0), cmap='gray')
    # plt.show()
    # exit()
    return state, noise


def learn():
    '''
    This is the optimizer for the training
    '''
    # sample from memory
    obses_t, actions, rewards, obses_tp1, dones = buffer.sample(BATCH_SIZE)
    
    # make a list of tensors adapted to GPU
    actions = [a.to(device) for a in actions]
    rewards = [r.to(device) for r in rewards]

    # mask of non final states
    non_final_mask = torch.tensor([True if o is not None else False for o in obses_tp1], device=device)
    # get non final states
    non_final_next_states = torch.cat([o for o in obses_tp1 if o is not None]).to(device)

    state_batch = torch.cat(obses_t).to(device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # compute the Q-values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # compute the loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    # update weights
    optimizer.step()


def update_target(model, target):
    target.load_state_dict(model.state_dict())


def train():
    episode_rewards = [0.0]

    obs = env.reset()
    state = get_state(obs)

    # times for logging
    train_start = datetime.datetime.now()
    episode_start = train_start

    for steps_done in range(STEPS):

        # if we want to render/show the game being played
        if args.render:
            env.render()

        # select an action with decaying epsilon greedy
        action = select_action(state, steps_done)
        
        # take one action and observe the outcome
        obs, reward, done, info = env.step(action)
        
        episode_rewards[-1] += reward

        # check if the game has ended
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
            # time calculation for logging
            episode_end = datetime.datetime.now()
            episode_time = episode_end - episode_start
            tot_time = episode_end - train_start
            episode_start = episode_end

            # average reward of the last 50 epochs
            avg_50 = np.round(np.mean(episode_rewards[-50:]), 1)

            # logs (print, log file, wandb)
            print(f'Total steps: {steps_done} \t Episodes: {len(episode_rewards)} \t Time: {tot_time} \t Episode Time: {episode_time} \t Reward: {episode_rewards[-1]} \t Reward Avg (Last 50): {avg_50}')
            log.info(f'Total steps: {steps_done} \t Episodes: {len(episode_rewards)} \t Time: {tot_time} \t Episode Time: {episode_time} \t Reward: {episode_rewards[-1]} \t Reward Avg (Last 50): {avg_50}')
            wandb.log({'reward': episode_rewards[-1], 'reward avg (last 50)': avg_50})

            # reset environment for next episode
            obs = env.reset()
            state = get_state(obs)
            episode_rewards.append(0.0)
        
        if steps_done > INITIAL_MEMORY:
            if steps_done % LEARNING_FREQ == 0:
                # optimize
                learn()

            if steps_done % TARGET_UPDATE == 0:
                # update target model
                update_target(policy_net, target_net)

        # save model checkpoint
        if steps_done % SAVE_FREQ == 0:
            time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            torch.save(policy_net.state_dict(), models_dir + time + '_' + str(steps_done))

    # close the environment
    env.close()


def test():
    # time for logging
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rewards = []

    for episode in range(TEST_EP):

        if args.video:
            states = []
            if args.attack:
                attacks = []
                # print(env.observation_space.shape)
                # exit()

        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        done = False

        while not done:
            # check if we are trying to attack and which attack
            if args.attack == 'noise':
                state, attack = rand_noise(state)
            elif args.attack == 'deepfool':
                attack, _, _, _, state = deepfool(state, policy_net, num_actions=num_actions)
            elif args.attack == 'simba':
                action = policy_net(state.to('cuda')).max(1)[1].view(1,1)
                state, attack = simba(state, action, policy_net)
            elif args.attack == 'simba_mod':
                action = policy_net(state.to('cuda')).max(1)[1].view(1,1)
                state, attack = simba_mod(state, action, policy_net)

            if args.video:
                states.append(state.to('cpu').reshape(1, -1, 84).permute(1, 2, 0).numpy())
                if args.imshow:
                    plt.imshow(state.to('cpu').reshape(1, -1, 84).permute(1, 2, 0), cmap='gray')
                    plt.show()
                    cv2.imshow('image', state.to('cpu').reshape(1, -1, 84).permute(1, 2, 0).numpy())
                    cv2.waitKey(0)

                if args.attack:
                    if args.attack == 'deepfool':
                        attacks.append(np.transpose(attack.reshape(1, -1, 84), (1, 2, 0)))
                        if args.imshow:
                            plt.imshow(np.transpose(attack.reshape(1, -1, 84), (1, 2, 0)), cmap='gray')
                            plt.show()
                            cv2.imshow('image', np.transpose(attack.reshape(1, -1, 84), (1, 2, 0)))
                            cv2.waitKey(0)  
                    else:
                        attacks.append(attack.to('cpu').reshape(1, -1, 84).permute(1, 2, 0).numpy())
                        if args.imshow:
                            plt.imshow(attack.to('cpu').reshape(1, -1, 84).permute(1, 2, 0).numpy(), cmap='gray')
                            plt.show()
                            cv2.imshow('image', attack.to('cpu').reshape(1, -1, 84).permute(1, 2, 0).numpy())
                            cv2.waitKey(0)

            action = policy_net(state.to('cuda')).max(1)[1].view(1,1)

            if args.render:
                env.render()

            # take one action and observe the outcome
            obs, reward, done, info = env.step(action)

            total_reward += reward

            # check if the game has ended
            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

        # log results of episode
        rewards.append(total_reward)
        print("Finished Episode {} with reward {}".format(episode, total_reward))
        log.info("Finished Episode {} with reward {}".format(episode, total_reward))

        # video
        if args.video:
            width = env.observation_space.shape[0]
            height = env.observation_space.shape[1]

            states = np.array(states)
            states = np.repeat(states, 3, axis=3)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_dir + '/' + str(episode) + '.mp4', fourcc, float(FPS), (width, height))

            for frame in states:
                video.write(np.uint8(np.clip(frame[:width, :height] * 255.0, 0, 255)))
            video.release()

            if args.attack:
                attacks = np.array(attacks)
                attacks = np.repeat(attacks, 3, axis=3)

                video = cv2.VideoWriter(video_dir + '/' + str(episode) + '_' + args.attack + '.mp4', fourcc, float(FPS), (width, height))

                for frame in attacks:
                    video.write(np.uint8(np.clip(frame[:width, :height] * 255.0, 0, 255)))

                video.release()

                scaled_attacks = (attacks - attacks.min()) / (attacks.max() - attacks.min())
                scaled_video = cv2.VideoWriter(video_dir + '/' + str(episode) + '_' + args.attack + '_scaled.mp4', fourcc, float(FPS), (width, height))

                for frame in scaled_attacks:
                    scaled_video.write(np.uint8(np.clip(frame[:width, :height] * 255.0, 0, 255)))
                scaled_video.release
                

    # log results of testing
    avg_reward = np.mean(rewards)
    print("Finished Testing {} Episodes with average reward {}".format(TEST_EP, avg_reward))
    log.info("Finished Testing {} Episodes with average reward {}".format(TEST_EP, avg_reward))

    # close the environment
    env.close()


if __name__ == "__main__":

    args = parse_args()

    game = args.game

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    models_dir = '../../models/' + game + '/'
    log_dir = '../../logs/' + game + '/'
    video_dir = log_dir + time + '_videos'

    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)

    log.basicConfig(filename=log_dir+time+'.log', level=log.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    print(f'Logging on {log_dir+time+".log"}')

    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.01
    EPS_DECAY = 200000
    TARGET_UPDATE = 1000
    LEARNING_RATE = 1e-4
    LEARNING_FREQ = 4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY
    STEPS = int(2e7)
    SAVE_FREQ = 10000
    TEST_EP = 10
    FPS = 30

    log.info(f'BATCH_SIZE: {BATCH_SIZE}')
    log.info(f'GAMMA: {GAMMA}')
    log.info(f'EPS_START: {EPS_START}')
    log.info(f'EPS_END: {EPS_END}')
    log.info(f'EPS_DECAY: {EPS_DECAY}')
    log.info(f'TARGET_UPDATE: {TARGET_UPDATE}')
    log.info(f'LEARNING_RATE: {BATCH_SIZE}')
    log.info(f'LEARNING_FREQ: {LEARNING_FREQ}')
    log.info(f'INITIAL_MEMORY: {INITIAL_MEMORY}')
    log.info(f'MEMORY_SIZE: {MEMORY_SIZE}')
    log.info(f'STEPS: {STEPS}')
    log.info(f'SAVE_FREQ: {SAVE_FREQ}')
    log.info(f'TEST_EP: {TEST_EP}')

    if args.train:
        wandb.init(project='dqn_' + game)
        wandb.run.name = time

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    log.info(f'Using device: {device}')

    # define the environment
    env = make_atari(game.capitalize() + 'NoFrameskip-v4')
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    env = gym.wrappers.Monitor(env, log_dir + time + '_videos')
    num_actions = env.action_space.n

    # agent and target
    policy_net = DQN(in_channels=4, num_actions=num_actions).to(device)
    target_net = DQN(in_channels=4, num_actions=num_actions).to(device)
    update_target(policy_net, target_net)

    # if there is a saved model
    if len(os.listdir(models_dir)) != 0 and args.train:
        print(f'Loading a model: {os.listdir(models_dir)[-1]}')
        log.info(f'Loading a model: {os.listdir(models_dir)[-1]}')
        policy_net.load_state_dict(torch.load(models_dir + os.listdir(models_dir)[-1]))
        policy_net.eval()
        update_target(policy_net, target_net)

    # optimizer
    optimizer = optim.RMSprop(policy_net.parameters(), lr=LEARNING_RATE)

    # instantiate replay buffer
    buffer = ReplayBuffer(MEMORY_SIZE)

    if args.train:
        # train
        log.info('Start training ...')
        train()
        # save trained model
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f'Saving a model: {os.listdir(models_dir)[-1]}')
        log.info(f'Saving a model: {os.listdir(models_dir)[-1]}')
        torch.save(policy_net.state_dict(), models_dir + time)
        wandb.run.save()

    if args.test:
        # load model
        test_dir = '../../models/for_testing/'
        model = test_dir + game
        print(f'Loading a model: ' + game)
        log.info(f'Loading a model: {game}')
        policy_net.load_state_dict(torch.load(model))
        policy_net.eval()
        # test
        log.info('Start testing ...')
        test()
        
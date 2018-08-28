#!/usr/bin/env python3

import argparse
import gym
import os
import time

try:
    import gym_minigrid
except ImportError:
    pass

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)

# Define agent

save_dir = utils.get_save_dir(args.model)
agent = utils.Agent(save_dir, env.observation_space, args.argmax)

# Run the agent

done = True

def labels_from_switches(switches):
    start_index = 0
    index = 0
    labels = []
    while index < len(switches):
        if switches[index] != 0:
            for option_index in range(start_index, index + 1):
                labels.append(switches[index] - 1)
            index += 1
            start_index = index
        else:
            index += 1
    return labels

def save_episode(episode, data_path):
    episode_uuid = uuid.uuid4().hex
    labels = labels_from_switches(episode['switches'])
    episode['labels'] = labels
    episode_path = os.path.join(data_path, episode_uuid)
    os.makedirs(episode_path)
    inputs = np.concatenate([episode['obs'][time_step]['image'].reshape(1, -1) for time_step in range(len(episode['obs']) - 1)])
    inputs_path = os.path.join(episode_path, 'inputs.npy')
    np.save(inputs_path, inputs)
    length = pd.DataFrame([inputs.shape[0]], columns=['length'])
    length_path = os.path.join(episode_path, 'length.csv')
    length.to_csv(length_path)
    targets = np.asarray(episode['actions']).reshape(-1, 1)
    targets_path = os.path.join(episode_path, 'targets.npy')
    np.save(targets_path, targets)
    states_trace = np.concatenate([np.expand_dims(state, axis=0) for state in episode['states']])
    states_trace_path = os.path.join(episode_path, 'states')
    os.makedirs(states_trace_path)
    for state_index in range(len(episode['states'])):
        state_file = '%04d' % state_index
        state_path = os.path.join(states_trace_path, state_file + '.png')
        state_img = Image.fromarray(episode['states'][state_index])
        state_resize = transforms.Resize((state_img.height // 4, state_img.width // 4))
        state_img = state_resize(state_img)
        misc.imsave(state_path, state_img)
    #np.save(states_trace_path, states_trace)
    ground_truth_trace = np.concatenate(episode['states'])
    ground_truth_trace = Image.fromarray(ground_truth_trace)
    resize = transforms.Resize((160 * len(episode['states']), 160))
    ground_truth_trace = resize(ground_truth_trace)
    ground_truth_trace_path = os.path.join(episode_path, 'ground_truth_trace.png')
    #misc.imsave(ground_truth_trace_path, ground_truth_trace)
    switches = [1 if episode['switches'][time_step] else 0 for time_step in range(len(episode['switches']))]
    switches_path = os.path.join(episode_path, 'switches.npy')
    switches = np.asarray(switches).reshape(-1, 1)
    np.save(switches_path, switches)
    label_color_map = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]]
    label_trace = []
    for time_step in range(len(labels)):
        label_square = np.zeros((160, 160, 3), dtype=np.uint8)
        for channel in range(3):
            channel_color = label_color_map[labels[time_step]][channel]
            label_square[:,:,channel] = channel_color
        label_trace.append(label_square)
    ground_truth_trace = transforms.ToTensor()(ground_truth_trace).permute(1, 2, 0).numpy()
    label_trace = np.concatenate(label_trace)
    label_trace = Image.fromarray(label_trace)
    label_trace = transforms.ToTensor()(label_trace).permute(1, 2, 0).numpy()
    #labeled_ground_truth_trace = np.concatenate([label_trace, ground_truth_trace], axis=1)
    #labeled_ground_truth_trace = (labeled_ground_truth_trace * 255 / np.max(labeled_ground_truth_trace)).astype('uint8')
    labels_path = os.path.join(episode_path, 'labels.npy')
    labels = np.asarray(labels).reshape(-1, 1)
    np.save(labels_path, labels)
    

import uuid
from scipy import misc
INFO_MAP = {'no_event': 0, 'open_box': 1, 'open_door': 2, 'get_key': 3, 'unlock_door': 4, 'episode_completed': 5}
data_path = os.path.join(gym_minigrid.__path__[0], 'saved_gym_minigrid_episodes', env.__class__.__name__)
import numpy as np
from PIL import Image
import pandas as pd
import pickle as pkl
from torchvision import transforms
if not os.path.isdir(data_path):
    os.makedirs(data_path)
for episode_index in range(640):
    done = False
    obs = env.reset()
    #print("Instr:", obs["mission"])
    episode = {}
    episode['obs'] = []
    episode['rewards'] = []
    episode['states'] = []
    episode['done'] = []
    episode['actions'] = []
    episode['switches'] = []
    episode['info'] = []
    episode['obs'].append(obs)
    episode['states'].append(env.render().getArray())


    while not done:
        #time.sleep(args.pause)
        #renderer = env.render("human")
    
        action = agent.get_action(obs)
        episode['actions'].append(action)
        obs, reward, done, info = env.step(action)
        episode['obs'].append(obs)
        episode['rewards'].append(reward)
        episode['done'].append(done)
        episode['info'].append(info)
        episode['states'].append(env.render().getArray())
        if len(info.keys()) > 1:
            raise SystemError
        elif len(info.keys()) == 0:
            info['no_event'] = True
        info_keys = list(info.keys())
        episode['switches'].append(INFO_MAP[info_keys[0]])
        agent.analyze_feedback(reward, done)
    
        #if renderer.window is None:
        #    break

    save_episode(episode, data_path)

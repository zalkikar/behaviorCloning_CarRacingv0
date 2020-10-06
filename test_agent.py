from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch
from model import pyTorchModel

def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    while True:
        
        # preprocess
        state = state[:-12,6:-6]
        state = np.dot(state[...,0:3], [0.299, 0.587, 0.114])
        state = state/255.0

        # get action
        agent.eval()
        tensor_state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0)
        tensor_action = agent(tensor_state)
        a = tensor_action.detach().numpy()[0]

        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    # load agent
    agent = pyTorchModel()
    agent.load_state_dict(torch.load("models/agent.pkl"))

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

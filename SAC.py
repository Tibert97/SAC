import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) #error only
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import math
import os
import copy
from torch.distributions import Normal
import time
import glob
import io
import pybullet
import Box2D
import pybullet_envs
import base64
import torch
from IPython.display import HTML
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd.variable import Variable
import torchvision
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
import csv
import pickle
from torch.utils import data
import warnings
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # check whether a GPU is available

from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

class Agent(nn.Module):
    def __init__(self,d_state,d_action,action_limit):
        super(Agent,self).__init__()
        self.lin1 = nn.Linear(d_state,300)
        nn.init.uniform_(self.lin1.weight,-0.001,0.001)
        nn.init.uniform_(self.lin1.bias,-0.001,0.001)
        self.norm1 = nn.LayerNorm(300)

        self.lin2 = nn.Linear(300,300)
        nn.init.uniform_(self.lin2.weight,-0.001,0.001)
        nn.init.uniform_(self.lin2.bias,-0.001,0.001)
        self.norm2 = nn.LayerNorm(300)



        self.out_mean = nn.Linear(300,d_action)
        nn.init.uniform_(self.out_mean.weight,-0.001,0.001)
        nn.init.uniform_(self.out_mean.bias,-0.001,0.001)
        self.out_logstd = nn.Linear(300,d_action)
        nn.init.uniform_(self.out_logstd.weight,-0.001,0.001)
        nn.init.uniform_(self.out_logstd.bias,-0.001,0.001)
        
        self.action_limit = action_limit

    def forward(self,x):
        x = self.lin1(x)
        x = self.norm1(x)
        x = F.relu(x)
        

        x = self.lin2(x)
        x = self.norm2(x)
        x = F.relu(x)
        

        mean = self.out_mean(x)
        log_std = self.out_logstd(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        normal = Normal(mean, std)

        sampled_action = normal.rsample()
        log_prob = torch.sum(normal.log_prob(sampled_action),dim=1)
        #correction of log prob based on spinning up implementation
        log_prob -= torch.sum((2*(np.log(2) - sampled_action - F.softplus(-2*sampled_action))),dim=1)
        
        return torch.tanh(sampled_action)*self.action_limit,log_prob,torch.tanh(mean)*self.action_limit

class A_Critic(nn.Module):
    def __init__(self,d_state,d_action):
        super(A_Critic,self).__init__()
        self.lin1 = nn.Linear(d_state+d_action,300)
        self.norm1 = nn.LayerNorm(300)
        nn.init.uniform_(self.lin1.weight,-0.001,0.001)
        nn.init.uniform_(self.lin1.bias,-0.001,0.001)

        self.lin2 = nn.Linear(300,300)
        self.norm2 = nn.LayerNorm(300)
        nn.init.uniform_(self.lin2.weight,-0.001,0.001)
        nn.init.uniform_(self.lin2.bias,-0.001,0.001)

        self.lin3 = nn.Linear(300,300)
        self.norm3 = nn.LayerNorm(300)
        nn.init.uniform_(self.lin3.weight,-0.001,0.001)
        nn.init.uniform_(self.lin3.bias,-0.001,0.001)


        self.out = nn.Linear(300,1)
        nn.init.uniform_(self.out.weight,-0.001,0.001)
        nn.init.uniform_(self.out.bias,-0.001,0.001)
        

    def forward(self,state,action):
        x = torch.cat((state,action),dim=1)

        x = self.lin1(x)
        x = self.norm1(x)
        x = F.relu(x)
        

        x = self.lin2(x)
        x = self.norm2(x)
        x = F.relu(x)
    
        return (self.out(x))



class ReplayBuffer(object):
  def __init__(self,capacity):
    self.capacity = capacity
    self.memory = list()
    self.position = 0
    self.changed_since_last_access = False
    self.means = None
    self.stds = None
  
  def add_sample(self,sample):
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.position] = torch.Tensor(sample).detach()
    self.position = (self.position + 1) % self.capacity
    self.changed_since_last_access = True


  def get_sample(self, batch_size):
    if self.changed_since_last_access:
      self.recompute_statistics()

    samples = torch.stack(random.sample(self.memory, batch_size)).detach()
    norm_samples = copy.deepcopy(samples)
    for i in range(samples.size()[-1]):
      norm_samples[:,i] -= self.means[i]
      norm_samples[:,i] /= self.stds[i]
    norm_samples[np.isnan(norm_samples)] = samples[np.isnan(norm_samples)]
    return samples

  def recompute_statistics(self):
    full_memory = torch.stack(self.memory)
    self.means = torch.mean(full_memory,dim=0)
    self.stds = torch.std(full_memory,dim=0)
    self.changed_since_last_access = False

  def normalise_observation(self,observation):
    if self.changed_since_last_access:
      self.recompute_statistics()

    norm_observation = copy.deepcopy(observation)
    for i in range(d_state):
      norm_observation[:,i] -= self.means[i]
      norm_observation[:,i] /= self.stds[i]
    norm_observation[np.isnan(norm_observation)] = observation[np.isnan(norm_observation)]
    return norm_observation
  def __len__(self):
    return len(self.memory)


"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""

#Source: https://colab.research.google.com/drive/1flu31ulJlgiRL1dnN2ir8wGh9p7Zij2t#scrollTo=dGEFMfDOzLen

def show_video():
  mp4list = glob.glob('video/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")
    

def wrap_env(env):
  env = Monitor(env, './video', force=True)
  return env

def delete_videos():
  mp4list = glob.glob('video/*.mp4')
  for f in mp4list:
    os.remove(f)

def environment_step(environment,obs,step_counter):  
  with torch.no_grad():
    agent_input = (torch.Tensor(obs).view(1,-1)).to(device)
    agent.eval()
    action,_,_ = agent(agent_input)
    agent.train()
    action = action[0].cpu().numpy()
    observation, reward, done, info = env.step(action) 
    env.render()
    step_counter += 1
    if done:
      if step_counter == episode_length:
        alive = 1
      else:
        alive = 0
      show_video()
      print(alive)
    else:
      alive = 1
    replay_buffer.add_sample(np.concatenate((obs[:d_state],action,observation[:d_state],[alive],[reward])))
    return observation,reward,done,step_counter



def compute_q_targets(samples):
  s = samples[:,:d_state]
  a = samples[:,d_state:d_state+d_action]
  s_new = samples[:,d_state+d_action:d_state+d_action+d_state]
  d = samples[:,d_state+d_action+d_state]
  r = samples[:,d_state+d_action+d_state+1]

  policy,log_prob,_ = agent(s_new)

  critic_1_output = a_critic_target_1(s_new,policy).squeeze()
  critic_2_output = a_critic_target_2(s_new,policy).squeeze()
  minimum_outputs = torch.min(critic_1_output,critic_2_output)
  targets = r + GAMMA * d * (minimum_outputs-alpha_entropy*log_prob)

  return targets

def update_q_network(targets,samples):
  s = samples[:,:d_state]
  a = samples[:,d_state:d_state+d_action]
  loss_fn = torch.nn.MSELoss()

  q_a_optimizer_1.zero_grad()
  predictions_1 = a_critic_1(s,a).squeeze()
  loss_1 = loss_fn(predictions_1,targets.detach())
  loss_1.backward()
  q_a_optimizer_1.step()

  q_a_optimizer_2.zero_grad()
  predictions_2 = a_critic_2(s,a).squeeze()
  loss_2 = loss_fn(predictions_2,targets.detach())
  loss_2.backward()
  q_a_optimizer_2.step()

  q_a_losses_1.append(loss_1.item())
  q_a_losses_2.append(loss_2.item())




def update_agent_network(samples):
  s = samples[:,:d_state]
  a_optimizer.zero_grad()

  actions, log_prob,_ = agent(s)
  q_1 = a_critic_1(s, actions)
  q_2 = a_critic_2(s, actions)
  q_min = torch.min(q_1,q_2)

  q_loss = torch.mean(-q_min)
  entropy_loss = torch.mean(alpha_entropy*log_prob)
  loss = q_loss+entropy_loss
  a_losses_q.append(q_loss.item())
  a_losses_entropy.append(entropy_loss.item())
  a_losses_overall.append(loss.item())

  loss.backward()
  a_optimizer.step()

def update_target_network(model,target_model,tau):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(
            target_param.data * tau + param.data * (1.0 - tau)
        )


def agent_training(replay_buffer, epochs):
    for epoch in range(epochs):
        sample= replay_buffer.get_sample(agent_batches).to(device)
        q_targets = compute_q_targets(sample)

        update_q_network(q_targets,sample)
        update_agent_network(sample)  
        update_target_network(a_critic_1,a_critic_target_1,tau)
        update_target_network(a_critic_2,a_critic_target_2,tau)

def compare_q_values():
  sample = replay_buffer.get_sample(1).to(device)
  with torch.no_grad():
    s = sample[:,:d_state]
    a = sample[:,d_state:d_state+d_action]
    a_critic_1.eval()
    a_critic_target_1.eval()
    a_critic_target_2.eval()
    agent.eval()
    predictions = a_critic_1(s,a).squeeze()
    targets = compute_q_targets(sample)
    print('Target:',targets, 'Predictions:',predictions)
    a_critic_1.train()
    a_critic_target_1.train()
    a_critic_target_2.train()
    agent.train()


def act_in_environment(env):
  with torch.no_grad():
    obs = env.reset()
    env.render()
    agent.eval()
    cum_reward = 0
    while True:
      agent_input = torch.Tensor(obs).view(1,-1).to(device)
      _,_,action = agent(agent_input)
      observation, reward, done, info = env.step(action.squeeze(dim=0).cpu()) 
      env.render()
      obs = observation
      cum_reward += reward
      if done:
        break
    agent.train()
    print(cum_reward)
    show_video()
    env.close()
    return cum_reward

def plot_policy_losses():
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(len(q_a_losses_1)), q_a_losses_1, 'g', label = 'Q1 Loss')
    ax2.plot(range(len(q_a_losses_2)), q_a_losses_2, 'b', label = 'Q2 Loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Q1 Loss")
    ax2.set_ylabel("Q2 Loss")
    plt.legend()
    plt.show()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.plot(range(len(a_losses_q)), a_losses_q,'y', label="Agent Q Loss")
    ax1.plot(range(len(a_losses_entropy)), a_losses_entropy,'m', label="Agent Entropy Loss")
    ax2.plot(range(len(a_losses_overall)), a_losses_overall,'r', label="Agent Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Entropy Loss")
    ax2.set_ylabel("Agent Loss")
    plt.legend()
    plt.show()
g_lr = 1e-3
c_lr = 1e-3
a_lr = 1e-3
q_a_lr = 1e-3
critic_steps = 5
gan_epochs_start = 2000
gan_epochs = 10
agent_epochs = 10
LAMBDA = 10
GAMMA = 0.99
alpha_entropy = 0.2
tau = 0.995
agent_batches = 64
batch_size_train = 64
buffer = list()
environment_name = 'Walker2DBulletEnv-v0'
env = gym.make(environment_name)
d_state = env.observation_space.shape[0]
d_action = env.action_space.shape[0]
d_noise = 1
action_limit_max = env.action_space.high[0]
action_limit_min = env.action_space.low[0]
episode_length = env._max_episode_steps

agent = Agent(d_state,d_action,action_limit_max).to(device)
a_critic_1 = A_Critic(d_state,d_action).to(device)
a_critic_target_1 = copy.deepcopy(a_critic_1)
a_critic_2 = A_Critic(d_state,d_action).to(device)
a_critic_target_2 = copy.deepcopy(a_critic_2)

replay_buffer = ReplayBuffer(1000000)
a_losses_q = list()
a_losses_entropy = list()
a_losses_overall = list()
q_a_losses_1 = list()
q_a_losses_2 = list()

#initialise the buffer

env = gym.make(environment_name)
d_state = env.observation_space.shape[0]
print(d_state)
d_action = env.action_space.shape[0]
episode_length = env._max_episode_steps
print(d_action)
while replay_buffer.__len__() < 500:
  start = env.reset()
  step_counter = 0
  observation = start
  while True:
  #  time.sleep(1./60.)
    action = env.action_space.sample()
    obs = observation
    observation, reward, done, info = env.step(action)
    step_counter += 1
    if done:
      if step_counter == episode_length:
        alive = 1
      else:
        alive = 0
    else:
      alive = 1
    replay_buffer.add_sample(np.concatenate((obs[:d_state],action,observation[:d_state],[alive],[reward])))
    if done: 
      break

print(replay_buffer.__len__())
env.close()

a_optimizer= optim.Adam(agent.parameters(), lr=a_lr)
q_a_optimizer_1 = optim.Adam(a_critic_1.parameters(), lr=q_a_lr)
q_a_optimizer_2 = optim.Adam(a_critic_2.parameters(), lr=q_a_lr)

env = wrap_env(gym.make(environment_name))
obs = env.reset()
env.render()
step_counter = 0
cum_reward = 0
episode_counter = 1
for i in range(int(1e6)):
    obs,reward,done,step_counter = environment_step(env,obs,step_counter)
    cum_reward += reward
    if i % 50 == 0:
      agent_training(replay_buffer, epochs = 50)
    if done:
      print('Episode number',episode_counter)
      print('Total reward of episode:', cum_reward)
      cum_reward = 0
      compare_q_values()
      plot_policy_losses()
      obs = env.reset()
      step_counter = 0
      env.render()
      episode_counter += 1
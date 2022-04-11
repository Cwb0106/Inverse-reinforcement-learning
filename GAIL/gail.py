import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal
import math
from ReplayBuffer import ReplayBuffer
import random
import pickle
from nets import Actor, Q_Critic, discriminator
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")




class gail(object):
	def __init__(
		self,
		env_with_Dead,
		state_dim,
		action_dim,
		max_action,
		gamma=0.99,
		hidden_size=128,
		a_lr=1e-4,
		c_lr=1e-4,
		batch_size = 128,
		disc_iter=5
	):

		self.action_dim = action_dim
		self.memory = ReplayBuffer(state_dim, action_dim)
		self.actor = Actor(state_dim, action_dim, hidden_size, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
		self.actor_target = copy.deepcopy(self.actor)

		self.q_critic = Q_Critic(state_dim, action_dim, hidden_size).to(device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		self.discriminator = discriminator(state_dim + action_dim).to(device)
		self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
		self.disc_loss_func = nn.BCELoss()

		self.file = open('./Expert_demo/arm_trj.pkl', 'rb')
		self.pool = pickle.load(self.file)


		self.env_with_Dead = env_with_Dead
		self.action_dim = action_dim
		self.max_action = max_action
		self.gamma = gamma
		self.disc_iter = disc_iter
		self.policy_noise = 0.2*max_action
		self.noise_clip = 0.5*max_action
		self.tau = 0.005
		self.batch_size = batch_size
		self.delay_counter = -1
		self.delay_freq = 1

	def get_reward(self, observation, action):
		observation = torch.FloatTensor(np.expand_dims(observation, 0))

		action_tensor = torch.FloatTensor(action).reshape(-1, self.action_dim)

		traj = torch.cat([observation, action_tensor], 1)
		reward = self.discriminator.forward(traj)
		reward = - reward.log()
		return reward.detach().item()


	def select_action(self, state):#only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			a = self.actor(state)
		return a.cpu().numpy().flatten()

	def train(self):
		self.delay_counter += 1
		with torch.no_grad():
			s, a, r, s_prime, dead_mask = self.memory.sample(self.batch_size)
			noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
			smoothed_target_a = (
					self.actor_target(s_prime) + noise  # Noisy on target action
			).clamp(-self.max_action, self.max_action)

		# Compute the target Q value
		target_Q1, target_Q2 = self.q_critic_target(s_prime, smoothed_target_a)
		target_Q = torch.min(target_Q1, target_Q2)
		'''DEAD OR NOT'''
		if self.env_with_Dead:
			target_Q = r + (1 - dead_mask) * self.gamma * target_Q  # env with dead
		else:
			target_Q = r + self.gamma * target_Q  # env without dead


		# Get current Q estimates
		current_Q1, current_Q2 = self.q_critic(s, a)

		# Compute critic loss
		q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the q_critic
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		if self.delay_counter == self.delay_freq:
			# Update Actor
			a_loss = -self.q_critic.Q1(s,self.actor(s)).mean()
			self.actor_optimizer.zero_grad()
			a_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			self.delay_counter = -1

	def discriminator_train(self):
		expert_batch = random.sample(self.pool, self.batch_size)
		expert_observations, expert_actions = zip(* expert_batch)
		expert_observations = np.vstack(expert_observations)


		expert_observations = torch.FloatTensor(expert_observations).to(device)

		expert_actions = torch.FloatTensor(expert_actions).unsqueeze(1).to(device)

		expert_actions = torch.reshape(expert_actions, (self.batch_size, -1))
		expert_trajs = torch.cat([expert_observations, expert_actions], 1)
		expert_labels = torch.FloatTensor(self.batch_size, 1).fill_(0.0).to(device)


		observations, actions, _, _, _ = self.memory.sample(self.batch_size)

		observations = torch.FloatTensor(observations)

		actions_dis = torch.FloatTensor(actions).to(device)

		trajs = torch.cat([observations, actions_dis], 1)
		labels = torch.FloatTensor(self.batch_size, 1).fill_(1.0).to(device)

		for _ in range(self.disc_iter):
			expert_loss = self.disc_loss_func(self.discriminator.forward(expert_trajs), expert_labels)
			current_loss = self.disc_loss_func(self.discriminator.forward(trajs), labels)

			loss = (expert_loss + current_loss) / 2
			self.discriminator_optimizer.zero_grad()
			loss.backward()
			self.discriminator_optimizer.step()







	def save(self):
		torch.save(self.actor.state_dict(), "gail_actor.pth")
		torch.save(self.q_critic.state_dict(), "gail_q_critic.pth")
		# torch.save(self.discriminator.state_dict(), "discriminator.pth")


	def load(self):
		self.actor.load_state_dict(torch.load("gail_actor.pth"))
		self.q_critic.load_state_dict(torch.load("gail_q_critic.pth"))
		# self.discriminator.load_state_dict(torch.load("discriminator.pth"))

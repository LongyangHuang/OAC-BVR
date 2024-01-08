import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)
		

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)
		


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class OAC_BVR(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=2.5,
		beta=0.2,
		behaviorvalue_timesteps=3e5,
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.betaactor = Actor(state_dim, action_dim, max_action).to(device)
		self.betaactor_target = copy.deepcopy(self.betaactor)
		self.betaactor_optimizer = torch.optim.Adam(self.betaactor.parameters(), lr=3e-4)

		self.betacritic = Critic(state_dim, action_dim).to(device)
		self.betacritic_target = copy.deepcopy(self.betacritic)
		self.betacritic_optimizer = torch.optim.Adam(self.betacritic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha
		self.beta = beta
		self.behaviorvalue_timesteps = behaviorvalue_timesteps
		self.total_it = 0
		


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	

	def train_beta(self, replay_buffer, batch_size=200):
		
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
						
		betaactor_loss = F.mse_loss(self.betaactor(state), action) 

		# Optimize the behavior_actor 
		self.betaactor_optimizer.zero_grad()
		betaactor_loss.backward()
		self.betaactor_optimizer.step()
	

	def train(self, replay_buffer, batch_size=200):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		
		if self.total_it < self.behaviorvalue_timesteps:

			with torch.no_grad():
			# Select action according to policy and add clipped noise
				noise = (
					torch.randn_like(action) * self.policy_noise
				).clamp(-self.noise_clip, self.noise_clip)

				betanext_action = (
					self.betaactor(next_state) + noise
				).clamp(-self.max_action, self.max_action)
				

				target_q1, _ = self.betacritic_target(next_state, betanext_action)
				target_q = reward + not_done * self.discount * target_q1
			
			# Get current behavior value estimates
	
			current_q1, _ = self.betacritic(state, action)

			betacritic_loss = F.mse_loss(current_q1, target_q)
			# Optimize the behavior critic
			self.betacritic_optimizer.zero_grad()
			betacritic_loss.backward()
			self.betacritic_optimizer.step()
		
		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise1 = (
					torch.randn_like(action) * self.policy_noise
				).clamp(-self.noise_clip, self.noise_clip)

			next_action = (
					self.actor_target(next_state) + noise1
				).clamp(-self.max_action, self.max_action)

				
				# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q= reward + not_done * self.discount * target_Q

			q,_ = self.betacritic_target(state,action)
											
			# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)
					
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)\
				+float(self.alpha)*F.mse_loss(current_Q1, q)+float(self.alpha)*F.mse_loss(current_Q2, q)
				
			# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		
			
		

			# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

				# Compute actor loss
			pi = self.actor(state)
			Q = self.critic.Q1(state, pi)
			lmbda = float(self.beta)/Q.abs().mean().detach()
				
			actor_loss = -lmbda  * Q.mean() + F.mse_loss(pi, action) 

				# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

				# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.betacritic.parameters(), self.betacritic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	
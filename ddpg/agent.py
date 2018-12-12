from models import Actor, Critic
from utils import hard_update, soft_update, OrnsteinUhlenbeckActionNoise
import torch
import torch.nn.functional as F
from torch import optim


class Agent:
    def __init__(self, state_dim, action_dim, action_lim, memory, gamma=.99, tau=1e-03):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.memory = memory
        self.gamma = gamma
        self.tau = tau
        self.noise = OrnsteinUhlenbeckActionNoise(action_dim)
        # self.epsilon = 1

        self.actor = Actor(state_dim, action_dim, action_lim)
        self.actor_target = Actor(state_dim, action_dim, action_lim)
        # self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-04)
        self.actor_optim = optim.SGD(self.actor.parameters(), lr=1e-04, momentum=0.9, nesterov=True)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        # self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-03)
        self.critic_optim = optim.SGD(self.critic.parameters(), lr=1e-03, momentum=.9, nesterov=True)
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        action = self.actor(state).detach().numpy() + self.noise.sample() * self.action_lim
        return action

    def remember(self, *args):
        self.memory.add(*args)

    def replay(self, batch_size):
        if 2000 > len(self.memory):
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)
        dones = torch.cat(dones)

        # update critic
        actions_target = self.actor_target(next_states)
        state_action_target = self.critic_target(next_states, actions_target).detach().squeeze()
        y_expected = rewards + (self.gamma * state_action_target * (1 - dones))
        y_predicted = self.critic(states, actions).squeeze()
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optim.zero_grad()
        loss_critic.backward()
        for param in self.critic.parameters():
            param.grad.data.clamp_(-1, 1)
        self.critic_optim.step()

        # update actor

        pred_action = self.actor(states)
        loss_actor = -self.critic(states, pred_action).sum()
        self.actor_optim.zero_grad()
        loss_actor.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optim.step()

        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor_target, self.tau)

        # self.noise.reset()
        # self.epsilon = max(1e-4, self.epsilon*.999)

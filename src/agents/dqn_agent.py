import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from src.agents.replay_buffer import ReplayBuffer
from src.model.q_network import QNetwork

class DQNAgent:

    def __init__(self,state_size,action_size):
        self.batch_size = 64
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.lr = 0.001
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network =QNetwork(state_size,action_size).to(self.device)
        self.target_q_network = QNetwork(state_size,action_size).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.memory = ReplayBuffer()
        self.update_target_every =100
        self.step_count = 0

    def select_action(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_val = self.q_network(state)
        return torch.argmax(q_val).item()

    def store(self,state,action,reward,next_state,done):
        self.memory.push(state,action,reward,next_state,done)

    def train(self):
        if len(self.memory) <self.batch_size:
            return

        states,actions,rewards,next_states,dones = self.memory.sample(batch_size=self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.q_network(states).gather(1,actions)

        with torch.no_grad():
            max_next_q = self.target_q_network(next_states).max(1)[0].unsqueeze(1)

        target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.MSELoss()(current_q,target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



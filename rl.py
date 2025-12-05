import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# =====================
# Hyperparameters
# =====================
STATE_SIZE = 11
ACTION_SIZE = 3

LR = 0.001
GAMMA = 0.99

MEMORY_SIZE = 50000
BATCH_SIZE = 64

TAU = 0.001  # soft-update factor

# N-step returns
N_STEP = 3

# PER hyperparams
PER_ALPHA = 0.6          # prioritization strength
PER_BETA_START = 0.4     # initial importance-sampling correction
PER_BETA_INCREMENT = 0.001
PER_EPSILON = 1e-5       # to avoid zero priority


# =====================
# Dueling DQN MODEL
# =====================
class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(STATE_SIZE, 128)
        self.fc2 = nn.Linear(128, 64)

        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Advantage stream A(s, a)
        self.adv_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, ACTION_SIZE),
        )

    def forward(self, x):
        # allow single state (shape [STATE_SIZE])
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        value = self.value_stream(x)          # [batch, 1]
        advantage = self.adv_stream(x)        # [batch, ACTION_SIZE]

        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + advantage - advantage_mean

        if q_values.shape[0] == 1:
            return q_values.squeeze(0)        # [ACTION_SIZE]
        return q_values


# =====================
# Prioritized Replay Buffer
# =====================
class PrioritizedReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, transition, priority: float = 1.0):
        # use max priority so new samples are likely to be seen
        max_priority = max(self.priorities, default=priority)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_priority

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        priorities = np.array(self.priorities, dtype=np.float32)
        scaled = np.power(priorities, PER_ALPHA)
        probs = scaled / scaled.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # importance-sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priority(self, idx: int, priority: float):
        self.priorities[idx] = priority


# =====================
# SNAKE AGENT — PER + N-step + Dueling Double DQN
# =====================
class SnakeAgent:
    def __init__(self):
        # Main & target networks
        self.model = DuelingDQN()
        self.target_model = DuelingDQN()
        self.update_target()

        # Optimizer & loss (per-sample)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.SmoothL1Loss(reduction="none")

        # PER buffer
        self.memory = PrioritizedReplayBuffer(MEMORY_SIZE)

        # N-step buffer
        self.n_step = N_STEP
        self.n_step_buffer = deque(maxlen=self.n_step)

        # Exploration (epsilon-greedy)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999

        # PER beta annealing
        self.beta = PER_BETA_START

    # ----------------------------
    # Target network sync
    # ----------------------------
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def soft_update(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

    # ----------------------------
    # N-step helpers
    # ----------------------------
    def _get_n_step_transition(self):
        """
        From the current n_step_buffer, build:
        (state_0, action_0, R^n, next_state_n, done_n, n_steps)
        where R^n = sum_{k=0}^{n_steps-1} gamma^k * r_k
        """
        R = 0.0
        n_steps = 0
        for idx, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            R += (GAMMA ** idx) * r
            n_steps += 1
            if d:
                break

        state_0, action_0, _, _, _ = self.n_step_buffer[0]
        next_state_n, done_n = self.n_step_buffer[n_steps - 1][3], self.n_step_buffer[n_steps - 1][4]

        return (state_0, action_0, R, next_state_n, done_n, n_steps)

    # ----------------------------
    # Store transition (called from train loop)
    # ----------------------------
    def remember(self, state, action, reward, next_state, done):
        """
        Public API used by train.py:
        remember(state, action, reward, next_state, done)

        Internally we build and push N-step returns into PER buffer.
        """
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # when we have N steps, push one combined transition
        if len(self.n_step_buffer) == self.n_step:
            transition = self._get_n_step_transition()
            self.memory.add(transition, priority=1.0 + PER_EPSILON)

        # if episode ends, flush remaining transitions
        if done:
            while len(self.n_step_buffer) > 0:
                transition = self._get_n_step_transition()
                self.memory.add(transition, priority=1.0 + PER_EPSILON)
                self.n_step_buffer.popleft()

    # ----------------------------
    # Action selection
    # ----------------------------
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)

        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_vals = self.model(state_tensor)
        return torch.argmax(q_vals).item()

    # ----------------------------
    # PER + N-step + Double DQN update
    # ----------------------------
    def replay(self):
        if len(self.memory.buffer) < BATCH_SIZE:
            return

        samples, indices, weights = self.memory.sample(BATCH_SIZE, self.beta)

        # unpack N-step transitions
        states = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32)
        actions = torch.tensor([s[1] for s in samples], dtype=torch.long)
        rewards = torch.tensor([s[2] for s in samples], dtype=torch.float32)  # R^n
        next_states = torch.tensor(np.array([s[3] for s in samples]), dtype=torch.float32)
        dones = torch.tensor([s[4] for s in samples], dtype=torch.float32)
        steps_n = torch.tensor([s[5] for s in samples], dtype=torch.float32)

        # current Q(s,a)
        q_values_all = self.model(states)
        current_q = q_values_all.gather(1, actions.unsqueeze(1)).squeeze()

        # Double DQN: main picks actions, target evaluates
        next_q_main = self.model(next_states)
        next_actions = torch.argmax(next_q_main, dim=1)

        next_q_target = self.target_model(next_states)
        next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze()

        # gamma^n for each transition
        gamma_n = torch.pow(torch.full_like(steps_n, GAMMA), steps_n)

        # N-step target: R^n + (1 - done_n) * gamma^n * Q(s_n, a*)
        target_q = rewards + (1 - dones) * gamma_n * next_q_values

        # PER-weighted loss
        weights = weights.to(current_q.device)
        per_sample_loss = self.criterion(current_q, target_q.detach())
        loss = (weights * per_sample_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update priorities using absolute TD error
        td_errors = torch.abs(current_q - target_q).detach()
        for idx, err in zip(indices, td_errors):
            self.memory.update_priority(idx, float(err.item() + PER_EPSILON))

        # soft-update target network
        self.soft_update()

        # anneal beta
        self.beta = min(1.0, self.beta + PER_BETA_INCREMENT)

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

import torch
from collections import deque
import random
import numpy as np
from stadium import PongEnvironment
from utils import cleanup_memory, load_checkpoint

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

class MemoryBuffer:
    def __init__(self, size: int = 100000):
        self.buffer = deque(maxlen=size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int, min_idx: int = 0):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        idx = np.random.choice(np.arange(min_idx, len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx], idx

def create_memory_buffer(size: int = 100000):
    """Creation of replay buffer without network predictions"""
    memory = MemoryBuffer(size)
    env = PongEnvironment(max_score=20)
    rand_idx = random.randint(0, 2) #int(np.random.choice([0, 1, 2], size=1, replace=False)[0])
    state = env.apply(rand_idx, render_flag=False)
    for i in range(size):
        if i % 4 == 0:
            action = random.randint(0, 2) #int(np.random.choice([0, 1, 2], size=1, replace=False)[0])
        state_p, reward, done = env.apply(action, render_flag=False)
        memory.add((state, action, reward, state_p, done))
        state = state_p
        if done:
            env.reset()
        if not (i + 1) % (size // 100):
            print(f"{i + 1} / {size} rows created in buffer")
    cleanup_memory(device)
    env.stop()
    return memory

def initialize_training(q, q_target, buffer_size, base_path):
    cp_episode, cp_memory, cp_epsilon = load_checkpoint(q, q_target, filename=f"{base_path}/checkpoints/latest.pth")
    if cp_memory is None or len(cp_memory.buffer) < buffer_size:
        memory = create_memory_buffer(size=buffer_size)
        episodes_start = 0
    else:
        memory = cp_memory
        episodes_start = cp_episode
    return memory, episodes_start, cp_epsilon
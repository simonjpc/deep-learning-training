import os
import gc
import pygame
import torch
import torch.nn.functional as F
from config import HYPERPARAMS
from model import QNetwork, QTargetNetwork
import numpy as np
import random

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

def compute_loss(logits, y):
    loss = F.mse_loss(logits, y)
    return loss

def print_memory_usage():
    import psutil
    process = psutil.Process()
    return f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB"

def cleanup_memory(device):
    # Clear PyTorch cache
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    
    gc.collect()
    pygame.event.pump()

def save_checkpoint(q, q_target, episode, epsilon, memory, filename):
    checkpoint = {
        "episode": episode,
        "q_network_state": q.state_dict(),
        "q_target_network_state": q_target.state_dict(),
        "q_network_optimizer": q.optimizer.state_dict(),
        "epsilon": epsilon,
        "memory_buffer": memory,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(q_network, q_target_network, filename):
    if not os.path.exists(filename):
        print("No checkpoint found")
        return 0, None, 1.0  # Return default values
    
    checkpoint = torch.load(filename, weights_only=False)
    q_network.load_state_dict(checkpoint["q_network_state"])
    q_target_network.load_state_dict(checkpoint["q_target_network_state"])
    return checkpoint["episode"], checkpoint.get("memory_buffer"), checkpoint["epsilon"]

def compute_avg_action_value(network, states, device):
    batch_size = 32
    max_values = []
    for i in range(0, len(states), batch_size):
        batch_states = states[i:i + batch_size].to(device)
        action_values = network.forward(batch_states)
        max_values.append(torch.max(action_values, dim=1, keepdim=True).values)
    max_values = torch.cat(max_values, dim=0)
    return round(max_values.mean().item(), 7)

def create_state_sequence(memory, idx):
    """Create a sequence of states from memory buffer"""
    while True:
        # Check if any of the first 3 transitions have done=True
        if any(memory.buffer[idx - i - 1][4] for i in range(3)):
            idx -= 1
            continue
        break

    states = []
    states_p = []
    action = memory.buffer[idx][1]
    reward = memory.buffer[idx][2]
    done = memory.buffer[idx][4]
    for i in range(4):
        states.append(memory.buffer[idx - i][0])
        states_p.append(memory.buffer[idx - i][3])
    return states[::-1], action, reward, states_p[::-1], done

# initialize stuff
def networks_init(device):
    image_size = HYPERPARAMS["image_size"]
    action_space = HYPERPARAMS["action_space"]
    channels = HYPERPARAMS["channels"]
    q = QNetwork(image_size, action_space, channels).to(device)
    q_target = QTargetNetwork(image_size, action_space, channels).to(device)
    return q, q_target

def random_state_init():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_eval_set(memory, nb_samples, min_idx=4):
    _, eval_idxs = memory.sample(nb_samples, min_idx)
    eval_states = []
    for eval_i in eval_idxs:
        eval_state, _, _, _, _ = create_state_sequence(memory, eval_i)
        eval_states.append(eval_state)
    eval_states = torch.tensor(np.array(eval_states), dtype=torch.float32).to(device)
    return eval_states
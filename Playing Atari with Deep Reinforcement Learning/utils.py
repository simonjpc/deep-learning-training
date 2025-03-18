import gc
import os
import torch
import pygame

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def compute_loss(logits, labels):
    exp_logits = torch.exp(logits - torch.max(logits))
    softmax = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
    loss = - torch.sum(labels * torch.log(softmax + 1e-6))
    return softmax, loss

def mse(labels, logits):
    return ((labels - logits) ** 2).mean()


def concat_states(current, incoming, max_history=4):
    if not isinstance(incoming, torch.Tensor):
        incoming = torch.tensor(incoming, dtype=torch.float32).unsqueeze(0)
    if len(current) == 0:
        return incoming
    if len(current) >= max_history:
        current = current[-(max_history-1):]
    
    result = torch.cat((current, incoming), dim=0)
    del current, incoming
    return result

def compute_prediction(reward, logits, gamma=0.99):
    return reward + gamma * torch.max(logits).item()

def save_checkpoint(q_network, q_target_network, episode, epsilon, buffer, last_idx, filename):
    checkpoint = {
        "episode": episode,
        "q_network_state": q_network.state_dict(),
        "q_target_network_state": q_target_network.state_dict(),
        "q_network_optimizer": q_network.optimizer.state_dict(),
        "epsilon": epsilon,
        "buffer": buffer,
        "last_idx": last_idx,
    }
    torch.save(checkpoint, filename)

def save_checkpoint_test(last_idx, filename):
    checkpoint = {
        "last_idx": last_idx,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(q_network, q_target_network, filename):
    if not os.path.exists(filename):
        print("No checkpoint found")
        return 0, None, -1, 1.0  # Return default values
    
    checkpoint = torch.load(filename)
    q_network.load_state_dict(checkpoint["q_network_state"])
    q_target_network.load_state_dict(checkpoint["q_target_network_state"])
    return checkpoint["episode"], checkpoint.get("buffer"), checkpoint.get("last_idx"), checkpoint["epsilon"]

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

def compute_avg_action_value(network, states):
    action_values = network.forward(states.to(device))
    max_values = torch.max(action_values, dim=1, keepdim=True).values
    return round(max_values.mean().item(), 6)

class CustomLRScheduler:
    def __init__(self, optimizer, factor=0.5, patience=1000, increase_factor=1.1, min_lr=1e-6, max_lr=1e-3, verbose=True):
        self.optimizer = optimizer
        self.factor = factor
        self.increase_factor = increase_factor
        self.patience = patience
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.verbose = verbose
        
        self.best_loss = None
        self.current_loss = None
        self.bad_epochs = 0
        self.good_epochs = 0

    def step(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
            return
        
        if loss < self.best_loss:
            self.good_epochs += 1
            self.bad_epochs = 0
            if self.good_epochs >= self.patience:
                self._increase_lr()
                self.good_epochs = 0
            self.best_loss = loss
        else:
            self.good_epochs = 0
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self._decrease_lr()
                self.bad_epochs = 0

    def _increase_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = min(old_lr * self.increase_factor, self.max_lr)
            param_group['lr'] = new_lr
            if self.verbose and new_lr != old_lr:
                print(f'Increasing learning rate: {old_lr:.2e} -> {new_lr:.2e}')

    def _decrease_lr(self):
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            if self.verbose and new_lr != old_lr:
                print(f'Decreasing learning rate: {old_lr:.2e} -> {new_lr:.2e}')

    def get_last_lr(self):
        return [
            param_group.get("lr") for param_group in self.optimizer.param_groups
        ]
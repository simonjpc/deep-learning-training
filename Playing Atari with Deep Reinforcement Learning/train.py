import random
import numpy as np
import torch
from stadium import PongEnvironment
from model import QNetwork, QTargetNetwork
from utils import (
    mse,
    concat_states,
    load_checkpoint,
    save_checkpoint,
    print_memory_usage,
    cleanup_memory,
    compute_avg_action_value
)
from datetime import datetime

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def build_memory_buffer(q_network, buffer_size=1e5):
    buffer_size = int(buffer_size)
    buffer = []
    env = PongEnvironment(max_score=5)
    state_p, reward, done = env.apply(render_flag=True)
    states_input = torch.tensor(())
    for idx, n in enumerate(range(buffer_size)):
        states_input = concat_states(states_input, state_p)
        if len(states_input) < 4:
            continue
        with torch.no_grad():
            _ = q_network.forward(states_input.unsqueeze(0).to(device))
        r_action = q_network.action_selection(epsilon=1).item()
        state_p, reward, done = env.apply(action=r_action, render_flag=True)
        states_input_p = concat_states(states_input, state_p)
        buffer.append((states_input, r_action, reward, states_input_p, done))
        if not (idx + 1) % (buffer_size // 50):
            print(f"{idx + 1}/{buffer_size} rows created in buffer")
    env.stop()
    return buffer, idx


if __name__ == "__main__":

    BASE_PATH = "Playing Atari with Deep Reinforcement Learning"
    TARGET_UPDATE_FREQ = 1500 # 2000
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY = 0.7
    REWARD_SCALE_FACTOR = 1
    k = 4
    q = QNetwork(4).to(device)
    batch_size = 32
    q_network = QNetwork(4, batch_size).to(device)
    q_target_network = QTargetNetwork(4, batch_size).to(device)
    eval_samples = 2000
    buffer_size = int(6e4)
    gamma = 0.99
    
    start_episode, buffer, last_idx, epsilon = load_checkpoint(q_network, q_target_network, filename=f"{BASE_PATH}/checkpoints/latest.pth")
    if buffer is None or len(buffer) < buffer_size - 4:
        buffer, last_idx = build_memory_buffer(q, buffer_size)

    eval_sample_idxs = torch.randint(0, len(buffer), (eval_samples,)).tolist()
    eval_states = torch.stack([buffer[i][1-1] for i in eval_sample_idxs]).to(device)

    env = PongEnvironment(max_score=5)
    states_input = torch.tensor(())
    state_p, reward, done = env.apply(render_flag=False)
    env.stop()

    M = buffer_size * 100
    epsilon = EPSILON_START
    env = PongEnvironment(max_score=20)
    for episode in range(M):

        epsilon = max(EPSILON_END, round(EPSILON_START - ((episode / M) * 0.9 / (3*EPSILON_END)), 2))
        
        # states preprocess cold start
        states_input = torch.tensor(())
        while len(states_input) < 4:
            state_p, reward, done = env.apply(render_flag=True)
            states_input = concat_states(states_input, state_p)

        q_network.batch_size = 1
        if episode == 0 or not episode % 4:
            _ = q_network.forward(states_input.unsqueeze(0).to(device))
            action = q_network.action_selection(epsilon=epsilon).item()
        state_p, reward, done = env.apply(action=action, render_flag=True)
        if done:
        #     print("Game Over! Restarting")
            env.reset()  # Reset for new game
        states_input_p = concat_states(states_input, state_p)
        buffer.append((states_input, action, reward, states_input_p, done))
        buffer = buffer[1:] # to keep the same buffer size

        buffer_sample_idxs = torch.randint(0, len(buffer), (batch_size,)).tolist()
        buffer_states_samples = torch.stack([buffer[i][1-1] for i in buffer_sample_idxs]).to(device)
        buffer_action_samples = torch.tensor([buffer[i][2-1] for i in buffer_sample_idxs], dtype=torch.long).to(device)
        buffer_reward_samples = torch.tensor([buffer[i][3-1] for i in buffer_sample_idxs], dtype=torch.float32).to(device)
        buffer_states_p_samples = torch.stack([buffer[i][4-1] for i in buffer_sample_idxs]).to(device)
        buffer_dones_samples = torch.tensor([buffer[i][5-1] for i in buffer_sample_idxs], dtype=torch.long).to(device)

        q_network.batch_size = batch_size
        _ = q_network.forward(buffer_states_samples)
        logits = q_network.logits_from_actions(buffer_action_samples.unsqueeze(-1))

        with torch.no_grad():
            q_target_logits = q_target_network.forward(buffer_states_p_samples)
            max_values = torch.max(q_target_logits, dim=1, keepdim=True).values
            buffer_reward_samples_resized = buffer_reward_samples.unsqueeze(-1)
            buffer_dones_resized = buffer_dones_samples.unsqueeze(-1)
            y = buffer_reward_samples_resized * REWARD_SCALE_FACTOR + (1 - buffer_dones_resized) * gamma * max_values

        loss = mse(y, logits)
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2.0).sum() for p in q_network.parameters())
        loss = loss + l2_lambda * l2_norm

        q_network.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(q_network.parameters(), clip_value=1.0)
        q_network.optimizer.step()


        if device.type == "mps":
            torch.cuda.empty_cache()

        q_network.scheduler.step(loss.item())
        if not (episode + 1) % 100:
            print(f"loss: {round(loss.item(), 7)} ... mem usage: {print_memory_usage()}", )
            cleanup_memory(device)

        if not (episode + 1) % TARGET_UPDATE_FREQ:
            current_datetime = datetime.now()
            now = current_datetime.strftime("%d-%m-%Y-%Hh%Mm%Ss")
            avg_reward = compute_avg_action_value(q_network, eval_states)
            save_checkpoint(q_network, q_target_network, episode, epsilon, buffer, last_idx, filename=f"{BASE_PATH}/checkpoints/latest.pth")
            # for terminal logging
            print(f"Episode {episode+1} / {M} executed")
            print(f"q_network logits range: {logits.min()} to {logits.max()}")
            print(f"q_target_network logits range: {q_target_logits.min()} to {q_target_logits.max()}")
            print("last lr: ", q_network.scheduler.get_last_lr())
            print("epsilon: ", epsilon)
            print("average reward: ", avg_reward)
            print("\t\tLOSS: ", round(loss.item(), 7))

            with torch.no_grad():
                q_target_network.filters_layer1.data.copy_(q_network.filters_layer1.data)
                q_target_network.filters_layer2.data.copy_(q_network.filters_layer2.data)
                q_target_network.bias_layer1.data.copy_(q_network.bias_layer1.data)
                q_target_network.bias_layer2.data.copy_(q_network.bias_layer2.data)
                q_target_network.linear_layer.data.copy_(q_network.linear_layer.data)
                q_target_network.final_layer.data.copy_(q_network.final_layer.data)
                q_target_network.linear_bias.data.copy_(q_network.linear_bias.data)
                q_target_network.final_bias.data.copy_(q_network.final_bias.data)
                
        del buffer_states_samples, buffer_action_samples
        del buffer_reward_samples, buffer_states_p_samples
        del loss, logits, q_target_logits, max_values, y
        try:
            none_var = None
        except KeyboardInterrupt:
            print("Training interrupted. Saving checkpoint ...")
            save_checkpoint(q_network, q_target_network, episode, epsilon, buffer, last_idx, filename=f"{BASE_PATH}/checkpoints/latest.pth")


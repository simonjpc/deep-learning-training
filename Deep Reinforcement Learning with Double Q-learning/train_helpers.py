import torch
from utils import create_state_sequence
import numpy as np
from memory import initialize_training
from utils import (
    compute_loss,
    cleanup_memory,
    save_checkpoint,
    compute_avg_action_value,
    compute_avg_action_value,
    print_memory_usage,
)
from utils import networks_init
from writer import write_board
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/ddqn_experiment_1')

def starters(device, hyperparams):
    buffer_size = hyperparams["memory_buffer_size"]
    q, q_target = networks_init(device)
    epsilon = hyperparams["epsilon_start"]
    memory, episodes_start, epsilon = initialize_training(q, q_target, buffer_size, hyperparams["base_path"])
    return memory, q, q_target, epsilon

def forward_episode(env, memory, q, epsilon, device, episode, render_flag=False):
    """Train the model for a single episode."""
    _, _, _, states_iter, _ = create_state_sequence(memory, -1)
    states = torch.tensor(np.array(states_iter), dtype=torch.float32).unsqueeze(0).to(device)
    action = 0
    if episode % 4 == 0:
        logits = q.forward(states)
        action = q.action_selection(epsilon, logits).item()
    state_p, reward, done = env.apply(action, render_flag=render_flag)
    if done:
        env.reset()
    memory.add((states_iter[-1], action, reward, state_p, done))
    return epsilon


def update_epsilon(episode, total_episodes):
    """Update epsilon for the epsilon-greedy policy."""
    if episode < 0.1 * total_episodes:
        return 1 - (0.9 * episode / (0.1 * total_episodes))
    return 0.1


def sample_memory(memory, batch_size, device):
    """Sample a batch from the memory buffer."""
    _, idxs = memory.sample(batch_size, min_idx=4)
    batch_states, batch_actions, batch_rewards, batch_states_p, batch_done = [], [], [], [], []
    for idx in idxs:
        single_states, single_action, single_reward, single_states_p, single_done = create_state_sequence(memory, idx)
        batch_states.append(single_states)
        batch_actions.append(single_action)
        batch_rewards.append(single_reward)
        batch_states_p.append(single_states_p)
        batch_done.append(single_done)
    return (
        torch.tensor(np.array(batch_states), dtype=torch.float32).to(device),
        torch.tensor(np.array(batch_actions), dtype=torch.long).to(device),
        torch.tensor(batch_rewards, dtype=torch.float32).unsqueeze(-1).to(device),
        torch.tensor(np.array(batch_states_p), dtype=torch.float32).to(device),
        torch.tensor(batch_done, dtype=torch.long).unsqueeze(-1).to(device),
    )


def compute_target_values(q, q_target, batch_states_p, batch_rewards, batch_done, gamma):
    """Compute target values for Double Q-learning."""
    with torch.no_grad():
        q_logits = q.forward(batch_states_p)
        q_actions = q.action_selection(0, q_logits)
        q_action_values = q_target.values_from_actions(q_logits, q_actions)
        y = batch_rewards + gamma * q_action_values * (1 - batch_done)
        return q_action_values, y


def update_target_network(q, q_target):
    """Update the target network with the weights of the main network."""
    with torch.no_grad():
        q_target.conv_layer1.data.copy_(q.conv_layer1.data)
        q_target.bias_layer1.data.copy_(q.bias_layer1.data)
        q_target.conv_layer2.data.copy_(q.conv_layer2.data)
        q_target.bias_layer2.data.copy_(q.bias_layer2.data)
        q_target.ffn_layer.data.copy_(q.ffn_layer.data)
        q_target.ffn_bias.data.copy_(q.ffn_bias.data)
        q_target.output_layer.data.copy_(q.output_layer.data)
        q_target.output_bias.data.copy_(q.output_bias.data)


def log_training_progress(episode, nb_episodes, loss, epsilon, avg_reward, logits, q_action_values):
    """Log training progress."""
    print(f"Episode {episode + 1} / {nb_episodes} executed")
    print(f"Q logits range: {logits.min()} to {logits.max()}")
    print(f"Q target logits range: {q_action_values.min()} to {q_action_values.max()}")
    print(f"Epsilon: {epsilon}")
    print(f"Average Reward: {avg_reward}")
    print(f"\t\tLoss: {round(loss.item(), 7)}")

def training_loop(env, memory, q, q_target, epsilon, device, HYPERPARAMS, eval_states):
    for episode in range(HYPERPARAMS["episodes"]):
        epsilon = forward_episode(env, memory, q, epsilon, device, episode)
        epsilon = update_epsilon(episode, HYPERPARAMS["episodes"])

        batch_states, batch_actions, batch_rewards, batch_states_p, batch_done = sample_memory(
            memory, HYPERPARAMS["batch_size"], device
        )

        logits = q.forward(batch_states)
        actions = q.action_selection(epsilon, logits)
        action_values = q.values_from_actions(logits, actions)

        q_target_action_values, y = compute_target_values(q, q_target, batch_states_p, batch_rewards, batch_done, HYPERPARAMS["gamma"])

        loss = compute_loss(action_values, y)
        q.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(q.parameters(), clip_value=1.0)
        q.optimizer.step()

        if device.type == "mps":
            torch.cuda.empty_cache()

        if (episode + 1) % HYPERPARAMS["eval_freq"] == 0:
            print(f"loss: {round(loss.item(), 7)} ... mem usage: {print_memory_usage()}", )
            cleanup_memory(device)

        if episode % HYPERPARAMS["target_network_update_freq"] == 0:
            avg_reward = compute_avg_action_value(q, eval_states, device)
            log_training_progress(episode, HYPERPARAMS["episodes"], loss, epsilon, avg_reward, logits, y)
            write_board(writer, loss, avg_reward, q, q_target, logits, action_values, q_target_action_values, y, epsilon, episode)
            update_target_network(q, q_target)
            save_checkpoint(q, q_target, episode, epsilon, memory, filename=f"{HYPERPARAMS['base_path']}/checkpoints/latest.pth")

        del batch_states, batch_actions, batch_rewards, batch_states_p, batch_done
        del loss, logits, y

        writer.flush()

        try:
            1
        except KeyboardInterrupt:
            print("Training interrupted, saving checkpoint ...")
            save_checkpoint(q, q_target, episode, epsilon, memory, filename=f"{HYPERPARAMS['base_path']}/checkpoints/latest.pth")
            print("Done")
            env.stop()
    
    writer.close()
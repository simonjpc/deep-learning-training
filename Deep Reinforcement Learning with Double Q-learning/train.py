import torch
from stadium import PongEnvironment
from config import HYPERPARAMS
from utils import (
    random_state_init,
    build_eval_set,
)
from train_helpers import starters, training_loop

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

# load states & initializations
random_state_init()
memory, q, q_target, epsilon = starters(device, HYPERPARAMS)

# build eval set
eval_states = build_eval_set(memory, HYPERPARAMS["eval_set"], min_idx=4)

# start pong environment
env = PongEnvironment(max_score=20)

# Main training loop
training_loop(env, memory, q, q_target, epsilon, device, HYPERPARAMS, eval_states)
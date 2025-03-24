import torch
import math
from data_handler import get_dataset
from tokenizer import SimpleTokenizer
from model import BasicTransformerNetwork, compute_loss, full_backprop, optimizer_step
from utils import gradient_clipping, global_grad_norm, check_for_nan_inf

# dataset
text_dataset = get_dataset()

# tokenizer
tokenizer = SimpleTokenizer(text_dataset)
encode = tokenizer.get_encode_fn()
decode = tokenizer.get_decode_fn()

# training loop

max_window = 30
batch_size = 32
nb_epochs = 5
optim_states = {
    "m": {},
    "v": {},
    "t": 1,
}
model_dim, nb_heads = 512, 4
transformer = BasicTransformerNetwork(vocab_size=len(tokenizer.vocab), model_dim=model_dim, nb_heads=nb_heads)

for iter in range(len(text_dataset) // batch_size):
    
    batch_inputs = []
    batch_pos = []
    batch_outputs = []

    for _ in range(batch_size):

        start_idx = int(torch.randint(0, len(text_dataset) - max_window - 1, (1,)))
        input_text = text_dataset[start_idx:start_idx + max_window] # x
        output_text = text_dataset[start_idx+1:start_idx + max_window + 1] # y
        input_ids = encode(input_text)
        pos_ids = list(range(len(input_ids)))
        output_ids = encode(output_text)

        batch_inputs.append(input_ids)
        batch_pos.append(pos_ids)
        batch_outputs.append(output_ids)

    batch_inputs = torch.tensor(batch_inputs, dtype=torch.long)  # [batch_size, seq_len]
    batch_pos = torch.tensor(batch_pos, dtype=torch.long)
    batch_outputs = torch.tensor(batch_outputs, dtype=torch.long)

    logits = transformer.forward(batch_inputs, batch_pos)

    logits = torch.clamp(logits, -5, 5)
    softmax, loss = compute_loss(logits, batch_outputs)

    grads = full_backprop(
        batch_inputs,
        batch_outputs,
        softmax,
        transformer.parameters,
        transformer.activations,
        transformer.cache,
        transformer.attn_projs,
        transformer.attn_proj_back,
    )

    check_for_nan_inf(grads)

    # Clip gradients
    grads = gradient_clipping(grads, max_norm=1.0)

    # Log global gradient norm
    global_norm = global_grad_norm(grads)

    transformer.parameters, optim_states = optimizer_step(
        grads, transformer.parameters, optim_states, lr=5e-7
    )

    if not iter % 200:
        print(f"Positional Encoding Magnitude: {transformer.position_embed_norm}")
        print(f"Logits Range: {logits.min()} to {logits.max()}")
        print(f"Softmax Range: {softmax.min()} to {softmax.max()}")
        print(f"Iteration {iter}: Global Grad Norm = {global_norm:.4f}")
        print()
        print("training loss: ", float(loss))
        print()
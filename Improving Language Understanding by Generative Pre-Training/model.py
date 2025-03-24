import torch
import torch.nn.init as init
class BasicTransformerNetwork:

    def __init__(self, vocab_size, model_dim, nb_heads):
        
        # parameters and activations tracker
        self.parameters = {"nb_heads": nb_heads}
        self.cache = {"dropout_p": 0.2, "model_dim": model_dim, "nb_heads": nb_heads}
        self.activations = {}

        # model vector sizes
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.nb_heads = nb_heads

        # layers
        self.attn_projs = {}
        attn_size = model_dim // nb_heads
        for n in range(self.nb_heads):
            self.attn_projs[f"k_proj_{n}"] = self.init_weights((model_dim, attn_size)) #torch.randn((model_dim, attn_size), dtype=torch.float32) * 0.01
            self.attn_projs[f"q_proj_{n}"] = self.init_weights((model_dim, attn_size)) #torch.randn((model_dim, attn_size), dtype=torch.float32) * 0.01
            self.attn_projs[f"v_proj_{n}"] = self.init_weights((model_dim, attn_size)) #torch.randn((model_dim, attn_size), dtype=torch.float32) * 0.01
        self.embedding = self.init_weights((vocab_size, model_dim)) # torch.randn((vocab_size, model_dim), dtype=torch.float32) * 0.01
        self.parameters["embedding"] = self.embedding
        self.attn_proj_back = self.init_weights((attn_size * nb_heads, model_dim)) #torch.randn((attn_size * nb_heads, model_dim), dtype=torch.float32)
        self.ff_layer = self.init_weights((model_dim, model_dim)) #torch.randn((model_dim, model_dim), dtype=torch.float32) * 0.01
        self.parameters["W1"] = self.ff_layer
        self.ff_layer2 = self.init_weights((model_dim, model_dim)) #torch.randn((model_dim, model_dim), dtype=torch.float32) * 0.01
        self.parameters["W2"] = self.ff_layer2
        self.ff_bias = torch.ones((1, model_dim), dtype=torch.float32) * 1e-5
        self.parameters["b1"] = self.ff_bias
        self.ff_bias2 = torch.ones((1, model_dim), dtype=torch.float32) * 1e-5
        self.parameters["b2"] = self.ff_bias2

    def positional_encoding(self, x_pos):
        """sinus non-trainable positional encoding to add with input"""
        batch_size, seq_len = x_pos.shape
        div_term = torch.exp(
            torch.arange(0, self.model_dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / self.model_dim)
        )

        pe = torch.zeros(batch_size, seq_len, self.model_dim, dtype=torch.float32)

        pe[:, :, 0::2] = torch.sin(x_pos.unsqueeze(-1) * div_term)  # sin for even indices
        pe[:, :, 1::2] = torch.cos(x_pos.unsqueeze(-1) * div_term)  # cos for odd indices
        return pe * 0.005
    
    def residual_connexion(self, current_layer, dragged_layer):
        """generic residual connexion operation"""
        return current_layer + dragged_layer

    def input_block(self, x, x_pos):
        """embedding input and positional encoding input"""
        input_embed = self.embedding[x]
        position_embed = self.positional_encoding(x_pos)
        self.position_embed_norm = torch.norm(position_embed)
        # print(f"Positional Encoding Magnitude: {torch.norm(position_embed)}")
        total_input = input_embed + position_embed
        self.cache["input_block_input"] = total_input
        total_input, total_input_mean, total_input_std = self.layernorm(total_input)
        self.cache["input_block_normalized_input"] = total_input
        self.cache["input_block_mean"] = total_input_mean
        self.cache["input_block_std"] = total_input_std
        total_input = self.dropout(total_input, 0.2)
        return total_input
    
    def attention_block(self, x):
        """full attention block, batch-compatible"""
        all_attn_outputs = []
        batch_size, seq_len, model_dim = x.shape
        head_size = model_dim // self.nb_heads

        self.cache["attn_X"] = x

        # Create lower triangular mask once and broadcast across batch
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float32))
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, seq_len]

        for n in range(self.nb_heads):
            # Project input to K, Q, V for this head
            K = torch.einsum("bsm,mh->bsh", x, self.attn_projs[f"k_proj_{n}"])
            Q = torch.einsum("bsm,mh->bsh", x, self.attn_projs[f"q_proj_{n}"])
            V = torch.einsum("bsm,mh->bsh", x, self.attn_projs[f"v_proj_{n}"])

            self.cache[f"K{n}"] = K
            self.cache[f"Q{n}"] = Q
            self.cache[f"V{n}"] = V

            # Compute attention scores: Q @ K.T (batched attention matmul)
            pre_attention = torch.einsum("bqh,bkh->bqk", Q, K)  # [batch, seq, seq]

            # Scale
            scale = pre_attention / torch.sqrt(torch.tensor(head_size, dtype=torch.float32))

            # Apply mask and handle -inf for softmax
            scale = scale.masked_fill(mask == 0, float("-inf"))

            # Softmax over the last dimension (attention scores over the sequence)
            attn_softm = torch.softmax(scale, dim=-1)

            attn_softm = self.dropout(attn_softm, 0.2)
            self.cache[f"attn_probs{n}"] = attn_softm

            # Compute attention output (weighted sum of values)
            attn_output = torch.einsum("bqk,bkh->bqh", attn_softm, V)  # [batch, seq, head_size]
            all_attn_outputs.append(attn_output)

        # Concatenate all head outputs along last dimension (feature dimension)
        all_attn_outputs = torch.cat(all_attn_outputs, dim=-1)  # [batch, seq, model_dim]

        # Final projection back to model dimension
        all_attn_outputs = torch.einsum("bsm,mh->bsh", all_attn_outputs, self.attn_proj_back)

        all_attn_outputs = self.dropout(all_attn_outputs, 0.2)
        return all_attn_outputs

    def linear_block(self, x):
        """linear layer block before output"""
        self.cache["linear_block_input"] = x
        x, x_mean, x_std = self.layernorm(x)
        self.cache["linear_block_normalized_input"] = x
        self.cache["linear_block_mean"] = x_mean
        self.cache["linear_block_std"] = x_std
        self.activations["A0"] = x
        ff_outout = x @ self.ff_layer + self.ff_bias # a0 * W1 + b1
        ff_outout = self.dropout(ff_outout, 0.2) # 
        ff_activation = self.relu_activation(ff_outout) # a1 = activation(W1 * x + b1)
        self.activations["A1"] = ff_activation
        ff_output2 = ff_activation @ self.ff_layer2 + self.ff_bias2 # a1 * W2  + b2
        logits = ff_output2 @ self.embedding.T
        return logits
    
    def dropout(self, x, p):
        mask = (torch.rand_like(x) > p).float()
        return x * mask
    
    def init_weights(self, shape):
        layer = torch.randn(shape, dtype=torch.float32)
        init.kaiming_normal_(layer, nonlinearity="relu")
        return layer

    def layernorm(self, x):
        """generic normalization layer"""
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        x_std = torch.std(x, dim=-1, keepdim=True)
        x_std = torch.clamp(x_std, min=1e-6)
        return (x - x_mean) / x_std, x_mean, x_std
    
    def relu_activation(self, x):
        """generic rectified linear unit layer"""
        return torch.relu(x)

    def forward(self, x, x_idx):
        """full transformer forward pass"""
        total_input = self.input_block(x, x_idx)
        attention = self.attention_block(total_input)
        residual1 = self.residual_connexion(attention, total_input)
        logits = self.linear_block(residual1)
        return logits
    

def compute_loss(logits, y):
    log_softmax = torch.log_softmax(logits, dim=-1)
    log_softmax_mean = log_softmax.mean()
    # log_softmax -= log_softmax_mean
    log_softmax /= 3
    target_log_probs = log_softmax.gather(2, y.unsqueeze(-1)).squeeze(-1)
    loss = -target_log_probs.mean()
    return log_softmax, loss

def layernorm_backprop(dout, x, mean, std):

    xmu = x - mean
    xsgma = 1 / (std + 1e-6)
    dnorm = dout * xsgma

    dx = dnorm
    dx -= dnorm.mean(dim=1, keepdims=True)
    dx -= (xmu / (std + 1e-6) ** 2) * (dnorm * xmu).mean(dim=1, keepdims=True)
    return dx

def full_backprop(x, y, output_softmax, parameters, activations, cache, attn_projs, attn_proj_back):
    grads = {}

    batch_size, seq_len = x.shape

    # ======= Linear Block Backprop (Logits to FFN) =======

    dZ2 = output_softmax.clone()
    dZ2[torch.arange(batch_size).unsqueeze(1), torch.arange(seq_len), y] -= 1
    dZ2 /= (batch_size * seq_len)  # Normalize across batch and sequence

    # Final projection (logits = ff_output2 @ embedding.T)
    grads["dEmbedding"] = torch.einsum("bsi,bsj->ij", dZ2, activations["A1"])

    # Backprop through FFN (W2, b2, ReLU, W1, b1)
    dA1 = torch.einsum("bsi,ij->bsj", dZ2, parameters["embedding"])

    grads["dW2"] = torch.einsum("bsi,bsj->ij", activations["A1"], dA1)
    grads["db2"] = dA1.sum(dim=(0, 1), keepdim=False).unsqueeze(0)

    dZ1 = dA1 * (activations["A0"] > 0).float()

    grads["dW1"] = torch.einsum("bsi,bsj->ij", activations["A0"], dZ1)
    grads["db1"] = dZ1.sum(dim=(0, 1), keepdim=False).unsqueeze(0)

    dA0 = torch.einsum("bsj,ij->bsi", dZ1, parameters["W1"])

    # Backprop through layernorm after attention block
    grads["dA0"] = layernorm_backprop(
        dA0,
        cache["linear_block_input"],
        cache["linear_block_mean"],
        cache["linear_block_std"],
    )

    # ======= Attention Block Backprop (Multi-head Attention) =======

    head_size = cache["model_dim"] // cache["nb_heads"]

    # Fix: Project back and reshape into heads directly
    combined_heads = torch.einsum("bsi,ij->bsj", grads["dA0"], attn_proj_back)
    dHeads = combined_heads.view(batch_size, seq_len, cache["nb_heads"], head_size).unbind(dim=2)

    dX = torch.zeros_like(cache["attn_X"])

    for h in range(cache["nb_heads"]):
        Q = cache[f"Q{h}"]
        K = cache[f"K{h}"]
        V = cache[f"V{h}"]
        attn_probs = cache[f"attn_probs{h}"]

        dHead = dHeads[h]  # [batch, seq, head_size]

        # Backprop into V
        dV = torch.einsum("bqk,bqh->bkh", attn_probs, dHead)
        grads[f"dW_v{h}"] = torch.einsum("bsm,bsh->mh", cache["attn_X"], dV)

        # Backprop into attention scores
        dAttn = torch.einsum("bqh,bkh->bqk", dHead, V)

        # Backprop through softmax (Jacobian trick)
        dScores = attn_probs * (dAttn - (dAttn * attn_probs).sum(dim=-1, keepdim=True))

        # Backprop into Q and K
        dQ = torch.einsum("bqk,bkh->bqh", dScores, K)
        dK = torch.einsum("bqk,bqh->bkh", dScores, Q)

        grads[f"dW_q{h}"] = torch.einsum("bsm,bsh->mh", cache["attn_X"], dQ)
        grads[f"dW_k{h}"] = torch.einsum("bsm,bsh->mh", cache["attn_X"], dK)

        # Backprop into input X
        dX += (
            torch.einsum("bsh,mh->bsm", dQ, attn_projs[f"q_proj_{h}"]) +
            torch.einsum("bsh,mh->bsm", dK, attn_projs[f"k_proj_{h}"]) +
            torch.einsum("bsh,mh->bsm", dV, attn_projs[f"v_proj_{h}"])
        )

    # ======= Residual Connection (Add attention & FFN gradients) =======
    dResidual = grads["dA0"] + dX

    # ======= Input Block Backprop (Embedding + Positional Encoding + Layernorm) =======
    dResidual /= (1 - cache["dropout_p"])  # Backprop through dropout

    assert dResidual.shape == dA0.shape
    
    dTotalInput = layernorm_backprop(
        dResidual,
        cache["input_block_input"],
        cache["input_block_mean"],
        cache["input_block_std"],
    )

    dInputEmbed = dTotalInput.clone()

    # Combine gradients into the embedding table
    if "dEmbedding" not in grads:
        grads["dEmbedding"] = torch.zeros_like(parameters["embedding"])

    for b in range(batch_size):
        for t in range(seq_len):
            token_id = x[b, t]
            grads["dEmbedding"][token_id] += dInputEmbed[b, t]
    
    # Optional positional encoding backprop (unused unless trainable)
    grads["dPositionalEncoding"] = dTotalInput.clone()

    return grads

def optimizer_step(grads, parameters, optim_states, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.001):
    
    m = optim_states["m"]
    v = optim_states["v"]
    t = optim_states["t"]

    # Map parameters to their corresponding gradient names
    param_grad_map = {
        "W1": "dW1",
        "W2": "dW2",
        "b1": "db1",
        "b2": "db2",
        "embedding": "dEmbedding"
    }

    for h in range(parameters["nb_heads"]):  # Use passed-in `nb_heads`, don't read from parameters
        param_grad_map[f"q_proj_{h}"] = f"dW_q{h}"
        param_grad_map[f"k_proj_{h}"] = f"dW_k{h}"
        param_grad_map[f"v_proj_{h}"] = f"dW_v{h}"

    for param_name, grad_name in param_grad_map.items():
        if param_name not in parameters or grad_name not in grads:
            continue

        param = parameters[param_name]
        grad = grads[grad_name]

        if param_name not in m:
            m[param_name] = torch.zeros_like(param)
            v[param_name] = torch.zeros_like(param)

        # AdamW moment updates
        m[param_name] = beta1 * m[param_name] + (1 - beta1) * grad
        v[param_name] = beta2 * v[param_name] + (1 - beta2) * (grad ** 2)

        # Bias correction
        m_hat = m[param_name] / (1 - beta1 ** t)
        v_hat = v[param_name] / (1 - beta2 ** t)

        # Weight update (decoupled weight decay for non-embedding params)
        param_update = lr * m_hat / (torch.sqrt(v_hat) + eps)

        if param_name in ["embedding", "b1", "b2"]:
            param -= param_update  # No weight decay on embeddings
        else:
            param -= param_update + weight_decay * param  # AdamW-style weight decay

        parameters[param_name] = param  # Update parameter in-place

    t += 1

    optim_states["m"] = m
    optim_states["v"] = v
    optim_states["t"] = t

    return parameters, optim_states
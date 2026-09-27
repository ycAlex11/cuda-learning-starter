
import torch 
import torch.nn.functional as F 


from common.extension_loader import load_cuda_extension
rmsnorm_cuda = load_cuda_extension("rmsnorm_cuda")

BATCH_SIZE = 1
SEQUENCE_LENGTH = 256
HIDDEN_SIZE = 896

Q_HEADS = 14
KV_HEADS = 2
HEAD_DIM = 64


MLP_INTERMEDIATE_SIZE = 4864
EPS = 1e-6
DTYPE = torch.float16

def main():
    torch.manual_seed(0)

    hidden_states = torch.randn(BATCH_SIZE,SEQUENCE_LENGTH,HIDDEN_SIZE,device="cuda",dtype=DTYPE)

    input_norm_weight = torch.ones(HIDDEN_SIZE,device="cuda",dtype=DTYPE,)

    post_attention_norm_weight = torch.ones(HIDDEN_SIZE,device="cuda",dtype=DTYPE,)

    q_proj = torch.nn.Linear(
        HIDDEN_SIZE,
        Q_HEADS * HEAD_DIM,
        bias=False,
        device="cuda",
        dtype=DTYPE,
    )

    k_proj = torch.nn.Linear(
        HIDDEN_SIZE,
        KV_HEADS * HEAD_DIM,
        bias=False,
        device="cuda",
        dtype=DTYPE,
    )

    v_proj = torch.nn.Linear(
        HIDDEN_SIZE,
        KV_HEADS * HEAD_DIM,
        bias=False,
        device="cuda",
        dtype=DTYPE,
    )

    o_proj = torch.nn.Linear(
        Q_HEADS * HEAD_DIM,
        HIDDEN_SIZE,
        bias=False,
        device="cuda",
        dtype=DTYPE,
    )

    normed_hidden_states = F.rms_norm(hidden_states,normalized_shape=(HIDDEN_SIZE,),weight=input_norm_weight,eps = EPS)

    q = q_proj(normed_hidden_states)
    k = k_proj(normed_hidden_states)
    v = v_proj(normed_hidden_states)

    print("Q shape:", q.shape)
    print("K shape:", k.shape)
    print("V shape:", v.shape)

    q = q.view(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        Q_HEADS,
        HEAD_DIM,
    ).transpose(1, 2)
    k = k.view(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        KV_HEADS,
        HEAD_DIM,
    ).transpose(1, 2)
    v = v.view(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        KV_HEADS,
        HEAD_DIM,
    ).transpose(1, 2)

    print("Q heads:", q.shape)
    print("K heads:", k.shape)
    print("V heads:", v.shape)
    KV_GROUP_SIZE = Q_HEADS // KV_HEADS
    k_for_q = k.repeat_interleave(KV_GROUP_SIZE, dim=1)
    v_for_q = v.repeat_interleave(KV_GROUP_SIZE, dim=1)

    print("K expanded for Q:", k_for_q.shape)
    print("V expanded for Q:", v_for_q.shape)

    scores = torch.matmul(q,k_for_q.transpose(-2,-1))
    scores = scores/(HEAD_DIM**0.5)
    print("Attention scores:", scores.shape)

    causal_mask = torch.triu(torch.ones(SEQUENCE_LENGTH,SEQUENCE_LENGTH,device="cuda",dtype=torch.bool),diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))

    attention_weights = torch.softmax(scores,dim = -1)
    print("Attention weights:", attention_weights.shape)
    print(
        "First row weight sum:",
        attention_weights[0, 0, 0].sum().item(),
    )

    attention_output = torch.matmul(attention_weights,v_for_q)
    print("Attention output:", attention_output.shape)

    attention_output = attention_output.transpose(1, 2).contiguous().view(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        HIDDEN_SIZE,
    )

    attention_output = o_proj(attention_output)

    print("Projected attention output:", attention_output.shape)

    residual_after_attention = hidden_states + attention_output
    print("After attention residual:", residual_after_attention.shape)
    post_attention_hidden = F.rms_norm(
        residual_after_attention,
        normalized_shape=(HIDDEN_SIZE,),
        weight=post_attention_norm_weight,
        eps=EPS,
    )

    print("After post-attention RMSNorm:", post_attention_hidden.shape)


    gate_proj = torch.nn.Linear(HIDDEN_SIZE,MLP_INTERMEDIATE_SIZE,bias = False, device="cuda",dtype=DTYPE)

    up_proj = torch.nn.Linear(HIDDEN_SIZE,MLP_INTERMEDIATE_SIZE,bias = False, device="cuda",dtype=DTYPE)
    down_proj =torch.nn.Linear(MLP_INTERMEDIATE_SIZE,HIDDEN_SIZE,bias = False, device="cuda",dtype=DTYPE)

    gate = gate_proj(post_attention_hidden)
    up = up_proj(post_attention_hidden)
    print("Gate:", gate.shape)
    print("Up:", up.shape)
    mlp_hidden = F.silu(gate) * up
    print("After SwiGLU:", mlp_hidden.shape)
    mlp_output = down_proj(mlp_hidden)

    print("MLP output:", mlp_output.shape)

    layer_output = residual_after_attention + mlp_output

    print("Final layer output:", layer_output.shape)


if __name__ == "__main__":
    main()

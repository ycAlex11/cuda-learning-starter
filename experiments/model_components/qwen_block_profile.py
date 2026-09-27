
import torch 
import torch.nn.functional as F



from common.extension_loader import load_cuda_extension
rmsnorm_cuda = load_cuda_extension("rmsnorm_cuda")
from common.benchmark_reporting import benchmark


BATCH_SIZE = 1
SEQUENCE_LENGTH = 256
HIDDEN_SIZE = 896
Q_HEADS = 14
KV_HEADS = 2
HEAD_DIM = 64
KV_GROUP_SIZE = Q_HEADS // KV_HEADS
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
    gate_proj = torch.nn.Linear(HIDDEN_SIZE,MLP_INTERMEDIATE_SIZE,bias = False, device="cuda",dtype=DTYPE)

    up_proj = torch.nn.Linear(HIDDEN_SIZE,MLP_INTERMEDIATE_SIZE,bias = False, device="cuda",dtype=DTYPE)
    down_proj =torch.nn.Linear(MLP_INTERMEDIATE_SIZE,HIDDEN_SIZE,bias = False, device="cuda",dtype=DTYPE)

    causal_mask = torch.triu(
        torch.ones(
            SEQUENCE_LENGTH,
            SEQUENCE_LENGTH,
            device="cuda",
            dtype=torch.bool,
        ),
        diagonal=1,
    )
    @torch.inference_mode()
    def forward_pytorch():
        normed_hidden_states = F.rms_norm(
            hidden_states,
            normalized_shape=(HIDDEN_SIZE,),
            weight=input_norm_weight,
            eps=EPS,
        )

        with torch.profiler.record_function("QKV projections"):
            q = q_proj(normed_hidden_states).view(
                BATCH_SIZE, SEQUENCE_LENGTH, Q_HEADS, HEAD_DIM
            ).transpose(1, 2)

            k = k_proj(normed_hidden_states).view(
                BATCH_SIZE, SEQUENCE_LENGTH, KV_HEADS, HEAD_DIM
            ).transpose(1, 2)

            v = v_proj(normed_hidden_states).view(
                BATCH_SIZE, SEQUENCE_LENGTH, KV_HEADS, HEAD_DIM
            ).transpose(1, 2)

        k_for_q = k.repeat_interleave(KV_GROUP_SIZE, dim=1)
        v_for_q = v.repeat_interleave(KV_GROUP_SIZE, dim=1)

        scores = torch.matmul(q, k_for_q.transpose(-2, -1))
        scores = scores / (HEAD_DIM ** 0.5)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v_for_q)

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            HIDDEN_SIZE,
        )
        with torch.profiler.record_function(
            "O projection + attention residual"
        ):
            attention_output = o_proj(attention_output)
            residual_after_attention = hidden_states + attention_output

        post_attention_hidden = F.rms_norm(
            residual_after_attention,
            normalized_shape=(HIDDEN_SIZE,),
            weight=post_attention_norm_weight,
            eps=EPS,
        )

        with torch.profiler.record_function("MLP"):
            mlp_hidden = F.silu(gate_proj(post_attention_hidden)) * up_proj(
                post_attention_hidden
            )

            mlp_output = down_proj(mlp_hidden)
            layer_output = residual_after_attention + mlp_output
        return layer_output
    
    @torch.inference_mode()
    def forward_custom_rmsnorm():
        normed_hidden_states = rmsnorm_cuda.rmsnorm_forward(
            hidden_states,
            input_norm_weight,
            EPS,
        )

        q = q_proj(normed_hidden_states).view(
            BATCH_SIZE, SEQUENCE_LENGTH, Q_HEADS, HEAD_DIM
        ).transpose(1, 2)

        k = k_proj(normed_hidden_states).view(
            BATCH_SIZE, SEQUENCE_LENGTH, KV_HEADS, HEAD_DIM
        ).transpose(1, 2)

        v = v_proj(normed_hidden_states).view(
            BATCH_SIZE, SEQUENCE_LENGTH, KV_HEADS, HEAD_DIM
        ).transpose(1, 2)

        k_for_q = k.repeat_interleave(KV_GROUP_SIZE, dim=1)
        v_for_q = v.repeat_interleave(KV_GROUP_SIZE, dim=1)

        scores = torch.matmul(q, k_for_q.transpose(-2, -1))
        scores = scores / (HEAD_DIM ** 0.5)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v_for_q)

        attention_output = attention_output.transpose(1, 2).contiguous().view(
            BATCH_SIZE,
            SEQUENCE_LENGTH,
            HIDDEN_SIZE,
        )

        attention_output = o_proj(attention_output)
        residual_after_attention = hidden_states + attention_output

        post_attention_hidden = rmsnorm_cuda.rmsnorm_forward(
            residual_after_attention,
            post_attention_norm_weight,
            EPS,
        )

        mlp_hidden = F.silu(gate_proj(post_attention_hidden)) * up_proj(
            post_attention_hidden
        )

        mlp_output = down_proj(mlp_hidden)
        return residual_after_attention + mlp_output
    
    for _ in range(10):
        forward_pytorch()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
    ) as prof:
        with torch.profiler.record_function("PyTorch Qwen-shaped layer"):
            forward_pytorch()
        torch.cuda.synchronize()
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
        )
    )

    
if __name__ == "__main__":
    main()

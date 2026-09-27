
import torch
import torch.nn.functional as F


from common.benchmark_reporting import benchmark

BATCH_SIZE = 1
SEQUENCE_LENGTH = 256
HIDDEN_SIZE = 896
MLP_INTERMEDIATE_SIZE = 4864
DTYPE = torch.float16

def main():
    torch.manual_seed(0)

    x = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        HIDDEN_SIZE,
        device="cuda",
        dtype=DTYPE,
    )

    gate_proj = torch.nn.Linear(
        HIDDEN_SIZE,
        MLP_INTERMEDIATE_SIZE,
        bias=False,
        device="cuda",
        dtype=DTYPE,
    )

    up_proj = torch.nn.Linear(
        HIDDEN_SIZE,
        MLP_INTERMEDIATE_SIZE,
        bias=False,
        device="cuda",
        dtype=DTYPE,
    )

    down_proj = torch.nn.Linear(
        MLP_INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        bias=False,
        device="cuda",
        dtype=DTYPE,
    )

    gate_up_proj = torch.nn.Linear(
        HIDDEN_SIZE,
        2 * MLP_INTERMEDIATE_SIZE,
        bias=False,
        device="cuda",
        dtype=DTYPE,
    )

    with torch.no_grad():
        gate_up_proj.weight[:MLP_INTERMEDIATE_SIZE].copy_(
            gate_proj.weight
        )

        gate_up_proj.weight[MLP_INTERMEDIATE_SIZE:].copy_(
            up_proj.weight
        )

    @torch.inference_mode()
    def mlp_separate():
        gate = gate_proj(x)
        up = up_proj(x)

        mlp_hidden = F.silu(gate) * up
        return down_proj(mlp_hidden)
    
    
    @torch.inference_mode()
    def mlp_fused_gate_up():
        gate_up = gate_up_proj(x)

        gate, up = torch.split(
            gate_up,
            MLP_INTERMEDIATE_SIZE,
            dim=-1,
        )

        mlp_hidden = F.silu(gate) * up
        return down_proj(mlp_hidden)
    
    separate_output = mlp_separate()
    fused_output = mlp_fused_gate_up()

    max_abs_error = (separate_output - fused_output).abs().max().item()

    correct = torch.allclose(
        separate_output,
        fused_output,
        atol=1e-3,
        rtol=1e-3,
    )

    print(f"Correctness: {'PASS' if correct else 'FAIL'}")
    print(f"Maximum absolute error: {max_abs_error:.8f}")

    if not correct:
        raise RuntimeError("Fused gate/up MLP output does not match baseline")
    for round_index in range(3):
        print(f"\n===== Round {round_index + 1} =====")

        if round_index % 2 == 0:
            benchmark("Fused gate/up MLP", mlp_fused_gate_up)
            benchmark("Separate gate + up MLP", mlp_separate)
        else:
            benchmark("Separate gate + up MLP", mlp_separate)
            benchmark("Fused gate/up MLP", mlp_fused_gate_up)


if __name__ == "__main__":
    main()

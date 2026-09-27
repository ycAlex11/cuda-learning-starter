import torch 

def run_stack(layers, x_seed, use_residual):
    x = x_seed.detach().clone().requires_grad_(True)
    hidden = x
    for layer in layers:
        update = torch.tanh(layer(hidden))

        if use_residual:
            hidden = hidden + update
        else:
            hidden = update

    loss = hidden.sum()
    loss.backward()

    return hidden.detach().norm().item(), x.grad.norm().item()

def main():
    torch.manual_seed(0)

    device = "cuda"
    width = 64
    depth = 20

    layers = torch.nn.ModuleList(
        [
            torch.nn.Linear(width, width, bias=False, device=device)
            for _ in range(depth)
        ]
    )

    with torch.no_grad():
        for layer in layers:
            layer.weight.mul_(0.1)

    x_seed = torch.randn(1, width, device=device)

    plain_output_norm, plain_gradient_norm = run_stack(
        layers,
        x_seed,
        use_residual=False,
    )

    residual_output_norm, residual_gradient_norm = run_stack(
        layers,
        x_seed,
        use_residual=True,
    )

    print(f"Depth: {depth}")
    print()
    print("Without residual connection")
    print(f"Output norm: {plain_output_norm:.6e}")
    print(f"Input gradient norm: {plain_gradient_norm:.6e}")
    print()
    print("With residual connection")
    print(f"Output norm: {residual_output_norm:.6e}")
    print(f"Input gradient norm: {residual_gradient_norm:.6e}")

if __name__ == "__main__":
    main()

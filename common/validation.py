import torch 
def compare_tensors(reference:torch.Tensor,candidate:torch.Tensor,atol:float,rtol:float):

    correct = torch.allclose(reference,candidate,atol=atol,rtol=rtol)

    max_abs_error=(reference-candidate).abs().max().item()

    return correct,max_abs_error

def compare_next_token_outputs(
    reference_output,
    candidate_output,
    atol: float,
    rtol: float,
) -> tuple[bool, bool, float]:
    reference_logits = reference_output.logits[:, -1, :]
    candidate_logits = candidate_output.logits[:, -1, :]

    correct, max_abs_error = compare_tensors(
        reference_logits,
        candidate_logits,
        atol=atol,
        rtol=rtol,
    )

    same_next_token = torch.equal(
        reference_logits.argmax(dim=-1),
        candidate_logits.argmax(dim=-1),
    )

    return correct, same_next_token, max_abs_error
# Benchmark Results

These are historical measurements and scenario validations, not a fresh run of the current source tree. Defaults below apply to the early CUDA experiments; individual sections specify different dtypes, shapes, or timing scopes. See [RUNNING.md](RUNNING.md) for current entry points and [ARCHITECTURE.md](ARCHITECTURE.md) for scope and known cleanup items.

## Environment

- GPU: NVIDIA GTX 1650 Max-Q, 4 GiB VRAM
- CUDA Toolkit: 12.4
- CUDA architecture: `sm_75`
- Data type: `float32`
- Input size: 1,000,000 values unless noted otherwise
- Timing method: CUDA Events with warm-up iterations and repeated measurements

## Phase 1 — Vector Add

Input:

```text
x + y
1,000,000 float32 values
```

| Implementation | Median latency | Estimated effective bandwidth |
| --- | ---: | ---: |
| PyTorch baseline | 0.0841 ms | 142.64 GB/s |
| Native CUDA kernel, 256 threads/block | 0.0842 ms | 142.59 GB/s |

Thread-block sweep for the native CUDA kernel:

| Threads per block | Median latency | Estimated effective bandwidth |
| --- | ---: | ---: |
| 128 | 0.0844 ms | 142.15 GB/s |
| 256 | 0.0842 ms | 142.59 GB/s |
| 512 | 0.0856 ms | 140.24 GB/s |

Conclusion: vector add is memory-bandwidth-bound on this GPU. The tested block sizes produced similar results; 256 threads per block was slightly better in this experiment.

## Phase 2 — CUDA Reduction

Input:

```text
sum(x)
1,000,000 float32 values, all values equal to 1.0
```

| Implementation | Median latency |
| --- | ---: |
| Initial shared-memory CUDA reduction | 0.1614 ms |
| Two elements per thread | 0.0896 ms |
| Warp-shuffle version | 0.0605 ms |
| Final optimized custom reduction | 0.0527 ms |
| PyTorch `torch.sum` | 0.0358 ms |

Conclusion: processing two elements per thread reduced the number of reduction blocks. Warp-level shuffle further reduced shared-memory and synchronization overhead. PyTorch remained faster because its reduction kernels and runtime are heavily optimized.

## Phase 3 — Fused PyTorch CUDA Extension

Input:

```text
x * scale + bias
4,194,304 float32 values
scale = 1.5
bias = 2.0
```

| Implementation | Stable median latency | Extra peak active memory |
| --- | ---: | ---: |
| PyTorch eager `x * scale + bias` | 0.4769 ms | 32 MiB |
| Fused CUDA extension `scale_add` | 0.2392 ms | 16 MiB |

Observed speedup:

```text
0.4769 / 0.2392 = about 1.99x
```

Conclusion: the fused extension eliminates the intermediate tensor from eager execution. It uses one CUDA kernel instead of two, reduces global-memory traffic, and halves the measured extra peak active memory.

## Phase 4 and 5 — Scheduler Validation

The scheduler experiments are functional system validations, not latency benchmarks.

### Real GPU Worker Task

| Item | Result |
| --- | --- |
| Operation | CUDA extension `scale_add` |
| Input size | 4,194,304 float32 values |
| Output check | `first_value = 3.5` |
| Worker-reported peak allocation | 32 MiB |
| Scheduler logical VRAM reservation | 64 MiB |
| Completion | succeeded |
| Logical VRAM after completion | 4096 MiB free |

### Worker Loss, Retry, and Stale Completion Validation

| Event | Result |
| --- | --- |
| Attempt 1 | Worker exited with code `1` without completion |
| Worker-loss handling | Task was requeued and 64 MiB logical reservation released |
| Attempt 2 | Replacement worker completed the real GPU task |
| Final task state | `succeeded` |
| Final attempt number | `2` |
| Stale attempt 1 completion | rejected |
| Final free logical VRAM | 4096 MiB |
| Retry-demo scheduled attempts | 2 |
| Retry-demo completed tasks | 1 |
| Retry-demo requeued tasks | 1 |
| Retry-demo rejected completions | 1 |

Conclusion: the scheduler uses at-least-once execution semantics. The task `attempt` number prevents stale completion reports from an old worker attempt from overwriting newer task state.

## Phase 6: Row-wise Softmax CUDA Extension

### Configuration

- GPU: NVIDIA GeForce GTX 1650 Max-Q, 4 GiB VRAM
- Input: `float32` CUDA tensor with shape `[4096, 256]`
- Warm-up iterations: 10
- Timed iterations: 100
- Timing method: CUDA Events
- Baseline: `torch.softmax(x, dim=1)`

### Correctness

- Maximum absolute error: `0.00000002`
- Row-sum minimum: `0.99999988`
- Row-sum maximum: `1.00000024`
- Result: PASS

### Latency Results

| Implementation | Median (ms) | Min / Max (ms) |
|---|---:|---:|
| PyTorch Softmax | 0.0614 | 0.0602 / 0.0728 |
| CUDA extension baseline | 0.3379 | 0.3364 / 0.3553 |
| CUDA extension with warp shuffle | 0.1884 | 0.1864 / 0.2069 |

### Memory

| Implementation | Extra peak active memory |
|---|---:|
| PyTorch Softmax | 4 MiB |
| CUDA extension baseline | 4 MiB |
| CUDA extension with warp shuffle | 4 MiB |

### Conclusion

The warp-shuffle version reduced custom CUDA Softmax median latency from 0.3379 ms to 0.1884 ms, a 1.79x speedup over the shared-memory baseline. Both custom kernels remain slower than PyTorch's optimized Softmax implementation. The optimization reduced block-wide shared-memory reductions and `__syncthreads()` calls by using register-level warp shuffle operations for the final reduction stage.

## Qwen-Shaped Attention Experiment

Hardware: NVIDIA GeForce GTX 1650 Max-Q, 4 GiB VRAM  
Attention configuration modeled after Qwen2.5-0.5B-Instruct:

```text
Batch size: 1
Sequence length: 256
Q heads: 14
KV heads: 2
Head dimension: 64
Hidden size: 896
```

The experiment creates random Q, K, and V tensors with the Qwen attention shape. It uses grouped-query attention: each of the 2 KV heads is shared by 7 Q heads.

Attention path:

```text
QK^T -> scale by sqrt(head_dim) -> Softmax -> weights x V
```

### FP32 CUDA Extension Integration

The custom CUDA Softmax extension accepts a 2D float32 tensor with 256 columns. The attention score tensor is reshaped:

```text
[1, 14, 256, 256] -> [3584, 256]
```

Each row represents one (batch, Q head, Q token) combination and contains scores for 256 K tokens.

Correctness:

- Custom CUDA Softmax vs PyTorch Softmax: PASS
- Full attention output vs PyTorch: PASS

Latency results:

| Operation | PyTorch FP32 | Custom CUDA extension FP32 |
| --- | ---: | ---: |
| Softmax only | 0.0543 ms | 0.1659 ms |
| Full attention | 0.3909 ms | 0.5018 ms |

The custom warp-shuffle Softmax is correct and integrates into a Qwen-shaped attention path, but PyTorch is faster for this workload. The full attention latency difference is approximately the same as the isolated Softmax latency difference, showing that the custom Softmax is the main source of additional latency.

### FP16 vs FP32 PyTorch Attention

| Metric | FP32 | FP16 |
| --- | ---: | ---: |
| PyTorch Softmax median | 0.0543 ms | 0.0407 ms |
| PyTorch full attention median | 0.3909 ms | 1.3302 ms |
| Attention extra peak active memory | 16.00 MiB | 12.06 MiB |

FP16 reduced Softmax latency and reduced the measured extra peak active memory by approximately 25%. However, FP16 full attention was slower on the GTX 1650 Max-Q.
The likely cause is hardware and kernel selection: GTX 1650 does not have Tensor Cores, so FP16 matrix multiplications do not receive the specialized matrix-multiply acceleration available on RTX-class GPUs. The two matrix multiplications in attention (QK^T and weights x V) dominate the full attention runtime.

### Conclusions

- The custom float32 CUDA Softmax extension was verified inside a Qwen-shaped grouped-query attention workload.
- Reshaping high-dimensional tensors into independent 2D Softmax rows is a practical extension-integration technique.
- FP16 reduces memory use, but does not guarantee lower latency.
- Performance depends on GPU architecture, operation type, tensor shape, and the selected low-level kernel.

Future work: add general sequence-length support and FP16 support to the custom extension, then profile with Nsight tools.

## Fused Scaled Softmax in Qwen-Shaped Attention

### Goal

Implement a custom CUDA fused scaled Softmax:

`Softmax(scores * scale)`

This combines scaling and Softmax into a single CUDA kernel launch.

### Environment

- GPU: NVIDIA GeForce GTX 1650 Max-Q, 4 GB VRAM
- CUDA Toolkit: 12.4
- Softmax input shape: `[4096, 256]`
- Attention shape: `[1, 14, 256, 64]`
- Attention dtype: float16
- Scale: `1 / sqrt(64) = 1 / 8`

### Correctness

Reference implementation:

`torch.softmax(x * scale, dim=1)`

| Dtype | Correctness | Maximum absolute error |
|---|---:|---:|
| float32 | PASS | 0.00000000 |
| float16 | PASS | 0.00000381 |

The FP16 kernel converts inputs to FP32 for max, exp, sum, and normalization, then converts the output back to FP16.

### Standalone Scaled Softmax Latency

| Dtype | Implementation | Median latency |
|---|---|---:|
| float32 | PyTorch separate scale + Softmax | 0.1207 ms |
| float32 | CUDA separate scale + custom Softmax | 0.2441 ms |
| float32 | CUDA fused scale + custom Softmax | 0.1886 ms |
| float16 | PyTorch separate scale + Softmax | 0.0769 ms |
| float16 | CUDA separate scale + custom Softmax | 0.2130 ms |
| float16 | CUDA fused scale + custom Softmax | 0.1825 ms |

### Fusion Result

| Dtype | CUDA separate | CUDA fused | Improvement |
|---|---:|---:|---:|
| float32 | 0.2441 ms | 0.1886 ms | 22.7% faster |
| float16 | 0.2130 ms | 0.1825 ms | 14.3% faster |

Fusion removes one separate scale kernel launch and its temporary scaled tensor. It avoids an extra global-memory write and read.

### Full FP16 Attention Result

PyTorch path:

`QK^T -> scale -> Softmax -> weightsV`

CUDA extension path:

`QK^T -> fused scale + Softmax -> weightsV`

| Implementation | Median latency |
|---|---:|
| PyTorch full attention | 1.3292 ms |
| CUDA extension full attention | 1.4060 ms |

Full attention correctness: `True`.

### Conclusion

Fusion improves the custom CUDA scaled Softmax in both FP32 and FP16. However, PyTorch remains faster overall because its Softmax and matrix multiplication kernels are more mature. In full attention, the two matrix multiplications remain the main cost, so optimizing Softmax alone has limited end-to-end impact.

This experiment demonstrates CUDA correctness validation, FP16 input/output with FP32 accumulation, shared-memory reduction, warp shuffle, kernel fusion, fair custom-kernel benchmarking, and integration into a Qwen-shaped attention path.

## Qwen-Shaped RMSNorm CUDA Extension

### Goal

Implement a custom CUDA RMSNorm extension for Qwen 2.5-0.5B shaped hidden states.

RMSNorm computes:

`output[i] = input[i] / sqrt(mean(input²) + eps) * weight[i]`

### Environment

- GPU: NVIDIA GeForce GTX 1650 Max-Q, 4 GB VRAM
- CUDA Toolkit: 12.4
- PyTorch: 2.6.0+cu124
- Input shape: `[1, 256, 896]`
- Hidden size: `896`
- Threads per block: `256`
- One CUDA block processes one token

### CUDA Design

Each block processes one token's 896 hidden values.

Each of the 256 threads uses a strided loop:

`col = tid, tid + 256, tid + 512, ...`

The kernel performs:

1. Each thread computes a local sum of squared input values.
2. Shared memory reduces 256 local sums.
3. Warp shuffle reduces the final 32 values.
4. The block obtains one RMS value for the token.
5. Each thread writes normalized values for its assigned columns.

FP16 implementation uses:

- FP16 input, weight, and output;
- FP32 conversion with `__half2float`;
- FP32 square accumulation, reduction, and RMS computation;
- FP16 output conversion with `__float2half`.

### Correctness

| Dtype | Correctness | Maximum absolute error | Tolerance |
|---|---:|---:|---|
| float32 | PASS | 0.00000191 | `atol=1e-5`, `rtol=1e-5` |
| float16 | PASS | 0.00781250 | `atol=1e-3`, `rtol=1e-3` |

### Latency Benchmark

| Dtype | Implementation | Median latency |
|---|---|---:|
| float32 | PyTorch `F.rms_norm` | 0.0694 ms |
| float32 | CUDA extension RMSNorm | 0.0274 ms |
| float16 | PyTorch `F.rms_norm` | 0.0927 ms |
| float16 | CUDA extension RMSNorm | 0.0202 ms |

### Result

| Dtype | Speedup |
|---|---:|
| float32 | CUDA extension is about 2.5x faster |
| float16 | CUDA extension is about 4.6x faster |

### Profiler Evidence

PyTorch Profiler showed that the FP16 PyTorch RMSNorm path performs multiple operations:

- `aten::pow`
- `aten::mean`
- `aten::add_`
- `aten::rsqrt`
- `aten::mul` twice
- `aten::copy_` twice

This indicates multiple CUDA operations and intermediate values.

The custom CUDA extension performs the RMSNorm calculation in one custom kernel, apart from output allocation:

- local square accumulation;
- shared-memory reduction;
- warp-shuffle reduction;
- RMS calculation;
- multiplication by weight;
- output writeback.

### Conclusion

For the fixed Qwen-shaped input `[1, 256, 896]` on GTX 1650 Max-Q, the specialized CUDA RMSNorm extension is faster than the current PyTorch baseline in both FP32 and FP16.

The profiler supports the measured result: PyTorch uses a general multi-operation RMSNorm path, while the custom extension fuses the computation into one CUDA kernel.

## Fused Residual + RMSNorm CUDA Extension

### Goal

Implement a fused CUDA kernel for:

`output = RMSNorm(input + residual, weight)`

The standard eager path is:

`combined = input + residual`

`output = RMSNorm(combined, weight)`

The fused CUDA kernel avoids materializing the intermediate `combined` tensor.

### Environment

- GPU: NVIDIA GeForce GTX 1650 Max-Q, 4 GB VRAM
- CUDA Toolkit: 12.4
- Input shape: `[1, 256, 896]`
- Hidden size: `896`
- Threads per block: `256`
- One block processes one token's 896 hidden values

### CUDA Design

Each block processes one token.

For each hidden value, the kernel computes:

`value = input[col] + residual[col]`

The kernel then:

1. Accumulates `value²` using a strided loop.
2. Reduces local sums with shared memory.
3. Uses warp shuffle for the final reduction.
4. Computes one RMS value for the token.
5. Writes `value / rms * weight[col]` to output.

FP16 input, residual, weight, and output are converted to FP32 for addition, accumulation, reduction, and RMS calculation. The final output is converted back to FP16.

### Correctness

| Dtype | Correctness | Maximum absolute error | Tolerance |
|---|---:|---:|---|
| float32 | PASS | 0.00000191 | `atol=1e-5`, `rtol=1e-5` |
| float16 | PASS | 0.00195312 | `atol=1e-3`, `rtol=1e-3` |

The FP16 reference uses FP32 addition and FP32 RMSNorm before converting the final result to FP16, matching the fused kernel's numerical path.

Current-code caveat: the correctness reference described above and the timing reference are not identical. The timing lambda in `experiments/cuda_operators/rmsnorm_test.py` uses eager addition in the input dtype. The historical measurements below are preserved, but this difference must be addressed before claiming a numerically matched comparison.

### Latency Benchmark

| Dtype | Implementation | Median latency |
|---|---|---:|
| float32 | PyTorch separate residual add + RMSNorm | 0.0891 ms |
| float32 | CUDA fused residual add + RMSNorm | 0.0338 ms |
| float16 | PyTorch separate residual add + RMSNorm | 0.1039 ms |
| float16 | CUDA fused residual add + RMSNorm | 0.0271 ms |

### Result

| Dtype | Speedup |
|---|---:|
| float32 | CUDA fused implementation is about 2.6x faster |
| float16 | CUDA fused implementation is about 3.8x faster |

Fusion removes the separate residual-add kernel and avoids the temporary tensor that would store `input + residual`.

### Residual Connection Gradient Demo

A 20-layer toy network was tested with deliberately small weights.

Without residual connection:

`hidden = update`

With residual connection:

`hidden = hidden + update`

Results:

| Configuration | Output norm | Input gradient norm |
|---|---:|---:|
| Without residual | 0.000000e+00 | 0.000000e+00 |
| With residual | 1.004672e+01 | 7.770072e+00 |

The plain 20-layer stack repeatedly replaces its hidden state with a small update, causing both the output and gradient to vanish. The residual stack preserves an identity path, so both information and gradients remain non-zero.

### Conclusion

Fused residual + RMSNorm improves performance by removing an intermediate tensor and combining residual addition, RMS reduction, normalization, weight multiplication, and output writeback into one CUDA kernel.

The experiment also demonstrates why residual connections are important in deep networks: they preserve forward information and provide a direct gradient path during backpropagation.

## Qwen-Shaped Decoder Layer: Profiling and MLP Fusion Experiment

### Setup

- GPU: NVIDIA GeForce GTX 1650 Max-Q, 4 GB VRAM
- Precision: float16
- Input shape: `[batch=1, sequence=256, hidden=896]`
- Qwen-aligned configuration:
  - 14 query heads, 2 KV heads, head dimension 64
  - MLP intermediate size: 4864
- The layer uses random weights with Qwen-compatible shapes. It is a structural microbenchmark, not real Qwen inference.

### Layer Structure

```text
RMSNorm
→ Q/K/V projections
→ grouped-query causal attention
→ output projection
→ residual connection
→ RMSNorm
→ SwiGLU MLP: gate/up/down projections
→ residual connection
```

### Custom RMSNorm Integration

The custom CUDA RMSNorm extension replaced both PyTorch RMSNorm calls in the full layer.

- Correctness: PASS
- Maximum absolute error: 0.001953125
- Comparison tolerance: atol=1e-2, rtol=1e-2

Stable full-layer measurements showed little end-to-end difference:

| Round | PyTorch layer | Custom RMSNorm layer |
| --- | ---: | ---: |
| 2 | 55.6044 ms | 55.2508 ms |
| 3 | 55.0894 ms | 54.3793 ms |

The custom RMSNorm kernel is substantially faster in isolation, but it affects only a small fraction of full-layer latency. The complete layer is dominated by matrix multiplication.

### PyTorch Profiler Result

Profiler measurements are used for relative hotspot analysis rather than absolute latency.

| Region | CUDA total | Approximate share of layer |
| --- | ---: | ---: |
| Full Qwen-shaped layer | 47.080 ms | 100% |
| SwiGLU MLP | 39.616 ms | 84.1% |
| QKV projections | 3.306 ms | 7.0% |
| O projection + attention residual | 2.177 ms | 4.6% |
| Attention batched matrix multiplications | 1.247 ms | 2.6% |
| Two RMSNorm calls | 0.293 ms | 0.6% |

The profiler recorded 7 aten::mm calls, corresponding to:

```text
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

The three MLP projections are the dominant hotspot. This is consistent with their larger dimensions:

```text
gate: 896 → 4864
up:   896 → 4864
down: 4864 → 896
```

### Gate/Up Projection Fusion Experiment

Tested an MLP optimization that concatenates the gate_proj and up_proj weights:

```text
Separate path:
896 → 4864  (gate)
896 → 4864  (up)

Fused gate/up path:
896 → 9728
→ split into gate [4864] and up [4864]
```

The fused layer copied the original gate weights into rows 0:4864 and up weights into rows 4864:9728, so both paths compute the same result.

- Correctness: PASS
- Maximum absolute error: 0.00000000

Stable timing results:

| Round | Separate gate + up MLP | Fused gate/up MLP |
| --- | ---: | ---: |
| 2 | 46.3310 ms | 46.4606 ms |
| 3 | 46.3330 ms | 46.4571 ms |

### Conclusion

Combining two projections into one larger GEMM did not produce a stable speedup on this GTX 1650 Max-Q.
The two approaches perform almost the same total matrix-multiplication work and produce the same amount of output data. Removing one GEMM launch is insignificant compared with roughly 46 ms of GEMM compute. cuBLAS already executes the separate 896 → 4864 GEMMs efficiently.
This experiment is a valid negative result: profiling identified MLP GEMMs as the hotspot, but this particular graph-level fusion did not improve latency. Future optimization directions should focus on approaches that reduce effective GEMM work or memory traffic, such as quantization, batching, or optimized inference runtimes.

## Qwen Worker Recovery Experiment

This section records the earlier generation-based retry experiment. The current `experiments/scheduling/qwen_retry_demo.py` uses prefill tasks and reports retained prefix-cache usage; see [PROGRESS.md](PROGRESS.md) for that later validation.

### Environment

- GPU: NVIDIA GeForce GTX 1650 Max-Q, 4 GiB VRAM
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Runtime: PyTorch CUDA, FP16 model weights
- Worker architecture: separate Python process with heartbeat thread
- Scheduler heartbeat timeout: 1 second
- Task: `qwen-retry-001`
- Prompt tokens: 9
- Generated tokens: 16
- Estimated KV Cache reservation: 1 MiB

### Result

| Step | Expected result | Observed result |
|---|---|---|
| Attempt 1 | Worker executes then exits before completion report | Worker exit code `1`; no completion result |
| Lost-worker handling | Task is requeued and reservation is released | Task became `pending`; free VRAM `3071 -> 3072 MiB` |
| Attempt 2 | New worker retries task successfully | Task became `succeeded`; attempts `2` |
| Stale completion | Old attempt 1 result is rejected | `False` |
| Duplicate completion | Repeated valid attempt 2 result is idempotent | `True`; free VRAM remained `3072 MiB` |

### Conclusion

The scheduler correctly handles a worker crash after inference but before result reporting. It reclaims the task's logical KV Cache reservation, retries the task on a restarted worker, rejects stale results with an old attempt number, and accepts duplicate valid completions without releasing the same reservation twice.

`used_vram_mb` and `free_vram_mb` are Scheduler-side logical accounting values for resource admission; they are not direct measurements of PyTorch/CUDA allocator memory.


## Qwen Request-Level Batching Integration

This experiment processes complete request batches sequentially on a persistent worker. It does not implement token-level continuous batching.

Hardware: NVIDIA GeForce GTX 1650 Max-Q, 4 GB VRAM  
Model: `Qwen/Qwen2.5-0.5B-Instruct`  
Precision: FP16  
Generation limit: `max_new_tokens=16`

Scheduler configuration:

| Setting | Value |
|---|---:|
| Logical worker VRAM | 4096 MiB |
| Static model reservation | 1024 MiB |
| Initial logical free VRAM | 3072 MiB |
| Reported CUDA free VRAM after model load | about 2304 MiB |
| Maximum batch size | 2 requests |
| Maximum batch KV-cache reservation | 64 MiB |

Normal request-batching result:

| Batch | Tasks | KV-cache reservation | Result |
|---|---|---:|---|
| `inference-batch-001` | `qwen-batch-001`, `qwen-batch-002` | 2 MiB | succeeded |
| `inference-batch-002` | `qwen-batch-003` | 1 MiB | succeeded |

The Qwen worker loaded the model once, processed both batches without reloading, and shut down only after all pending tasks were complete.

Failure-handling result:

| Batch | Tasks | Result |
|---|---|---|
| `inference-batch-001` | `qwen-batch-001`, `qwen-batch-002` | controlled failure; both tasks marked failed and reservation released |
| `inference-batch-002` | `qwen-batch-003` | succeeded |

Conclusion: the project now supports persistent GPU workers, queue-based batch execution, batch-level resource accounting, continuous processing of pending requests, and safe batch-level failure cleanup.

## Qwen Batch Policy Benchmark

Hardware: NVIDIA GeForce GTX 1650 Max-Q, 4 GB VRAM  
Model: `Qwen/Qwen2.5-0.5B-Instruct`  
Precision: FP16  
Requests per round: 3  
Generation limit: `max_new_tokens=16`  
Rounds per policy: 3  
Batch KV-cache budget: 64 MiB  
Batch wait limit: 50 ms  

| Policy | Median task latency | Median tail latency | Estimated throughput |
|---|---:|---:|---:|
| `max_batch_size=2` | 1281 ms | 2000 ms | 1.50 requests/s |
| `max_batch_size=4` | 1329 ms | 1329 ms | 2.26 requests/s |

Observations:

- With `max_batch_size=2`, the first two requests ran immediately as one batch; the third waited for the first batch to finish and became a second batch.
- With `max_batch_size=4`, three requests did not fill the batch, so the Scheduler intentionally waited about 62–63 ms before dispatching all three together.
- The larger batch increased median task latency by 48 ms, because of the intentional wait.
- The larger batch reduced median tail latency from 2000 ms to 1329 ms, a reduction of about 33.6%.
- Estimated throughput increased from 1.50 to 2.26 requests/s, about 1.51x.
- This demonstrates the serving trade-off: a short batching delay can improve throughput and tail latency when several requests arrive close together.

## Qwen Prefix Cache Reuse

Hardware: NVIDIA GeForce GTX 1650 with Max-Q Design (4 GB VRAM)  
Model: Qwen/Qwen2.5-0.5B-Instruct, FP16  
Experiment: two requests share a 951-token prefix. Request suffixes contain 6 and 5 tokens.

| Method | Median | Min / Max |
| --- | ---: | ---: |
| Full prefill for both requests | 13167.65 ms | 13150.88 / 13203.56 ms |
| Shared-prefix KV Cache reuse | 6845.17 ms | 6838.71 / 6873.64 ms |

Shared-prefix KV Cache reuse achieved a 1.92x prefill speedup. The full-prefill baseline recomputed the 951-token prefix for both requests. The cache path prefills the shared prefix once, then processes only each request-specific suffix.

Correctness was validated by comparing final-token logits from the full-prefill and cached paths. Under FP16, the maximum absolute logit error was 0.02343750 and 0.03320312 for the two requests; both paths selected the same next token.

Current implementation uses `copy.deepcopy` to isolate each request's cached prefix state. Production inference engines generally use block-based KV Cache sharing and copy-on-write instead of copying an entire cache.

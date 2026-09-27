# Running the Experiments

## Scope and Prerequisites

Run commands from the project root (`cuda-learning-starter`), not from inside an experiment subdirectory.

This guide describes the existing Windows setup. It is not yet a validated clean-machine installer. The examples span different stages of the project; shared helper changes can affect older scripts.

Local package metadata inspected during the documentation pass on 2026-09-26:

| Component | Observed version |
| --- | --- |
| Python | 3.11.0, project virtual environment |
| PyTorch | 2.6.0+cu124 |
| Transformers | 5.14.1 |
| NumPy | 2.4.6 |

This is an environment snapshot, not a dependency lock or a claim that every historical benchmark used these exact versions.

- CUDA programs/extensions require the NVIDIA driver, CUDA Toolkit, and MSVC x64 build tools.
- Build scripts target `sm_75` and assume Visual Studio 2022 Build Tools at its standard path. `build_cpp_cuda.ps1` also contains an explicit CUDA 12.4 path.
- Real-model experiments use `Qwen/Qwen2.5-0.5B-Instruct`. The initial load may download model/tokenizer files; subsequent loads can use the local Hugging Face cache.
- Logical Scheduler tests do not need a GPU or model weights.
- Gloo/DDP exercises use the installed PyTorch package but run on CPU.

## Select the Existing Python Environment

If PowerShell blocks activation scripts, the process-scoped policy below affects only the current shell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

All `python` commands below assume this environment is active. Alternatively, replace `python` with `.\.venv\Scripts\python.exe`.

Check the CUDA/PyTorch environment:

```powershell
python .\scripts\check_env.py
```

An environment-check failure is not evidence that the CPU-only budget test requires a GPU.

## 1. CUDA Fundamentals and Reduction

### Starter Vector Add

```powershell
.\scripts\build_cpp_cuda.ps1
.\build\vector_add.exe
python -m experiments.cuda_basics.vector_add.01_torch_vector_add
```

The build script compiles `experiments/cuda_basics/vector_add/vector_add_starter.cu`, not `experiments/cuda_basics/vector_add/02_vector_add.cu`. The latter is a separate learning experiment.

### Reduction Variants

The reduction CUDA files are standalone programs. Compile them with `nvcc` in a shell where the MSVC x64 compiler environment is available; activating the Python environment alone does not set up MSVC.

For example, from an x64 Visual Studio developer shell with `nvcc` on PATH:

```powershell
New-Item -ItemType Directory -Force .\build
nvcc -std=c++17 -O2 -arch=sm_75 --allow-unsupported-compiler .\experiments\cuda_basics\reduction\01_reduction.cu -o .\build\01_reduction.exe
.\build\01_reduction.exe
python -m experiments.cuda_basics.reduction.02_reduction
```

Repeat with the desired two-elements-per-thread or warp-shuffle source and a distinct output filename. Keep baseline and optimized executables separate. The unsupported-compiler flag is inherited from the local setup; it does not guarantee compatibility with arbitrary compiler versions.

## 2. PyTorch CUDA Extensions

Build the three modules using the project virtual environment:

```powershell
.\scripts\build_extension.ps1
.\scripts\build_softmax.ps1
.\scripts\build_rmsnorm.ps1
```

Rebuild verification note: during the migration check, the relocated Softmax build reached `nvcc` but failed to locate `cl.exe` in its nested build environment. Existing binaries were preserved and tested, but a fresh rebuild is not yet confirmed. See [STRUCTURE_CHANGES.md](STRUCTURE_CHANGES.md).

Then run selected correctness/timing experiments:

```powershell
python -m experiments.cuda_operators.vector_add_demo
python -m experiments.cuda_operators.softmax_test
python -m experiments.cuda_operators.softmax_fp16_test
python -m experiments.cuda_operators.softmax_scale_test
python -m experiments.cuda_operators.rmsnorm_test
python -m experiments.cuda_operators.rmsnorm_profile
```

Each operator has its own `cuda_extensions/<operator>/setup.py`. Build scripts copy native modules to `build/python/`, and `common.extension_loader` imports them from there. `common/benchmark_reporting.py` is now independent of Softmax, so unrelated timing consumers no longer require that extension.

Inspect correctness output before interpreting latency. Not every historical test raises an exception when a printed correctness check fails.

## 3. Synthetic Attention and Decoder-Layer Experiments

These use random tensors/weights, not the complete pretrained Qwen model:

```powershell
python -m experiments.model_components.attention_demo
python -m experiments.model_components.attention_fp16_baseline
python -m experiments.model_components.attention_fp16_extension
python -m experiments.model_components.residual_demo
python -m experiments.model_components.qwen_block_demo
python -m experiments.model_components.qwen_block_benchmark
python -m experiments.model_components.qwen_block_profile
python -m experiments.model_components.mlp_fusion_benchmark
```

Build the CUDA extensions first for paths that import them. Profiling is for locating expensive operations; use repeated benchmarks for latency comparisons. A faster isolated RMSNorm does not imply the same speedup for a whole decoder layer.

## 4. Real Qwen Inference and Cache Experiments

```powershell
python -m experiments.qwen_inference.qwen_inference
python -m experiments.qwen_inference.prefill_decode_demo
python -m experiments.qwen_inference.qwen_batch_demo
python -m experiments.qwen_inference.qwen_batch_benchmark
python -m experiments.qwen_inference.prefix_cache_demo
```

Read the selected script's input lengths, warm-up count, and run count before launching it. Long-prefix and uncached comparisons can take minutes. The first model load is not representative of warmed-up inference latency.

`kv_cache_demo.py` remains available as an earlier cache experiment. Its printed timing helper now comes from `common.benchmark_reporting`; this path does not require the Softmax extension.

For the cache-data-structure exercise without loading Qwen weights:

```powershell
python -m experiments.qwen_inference.prefix_cache_store_demo
```

This still imports the shared cache module and its Python dependencies. It is not a dependency-free test.

Prefix correctness checks compare final-position logits and next-token selection. They do not establish identical complete generated sequences.

## 5. Scheduler Scenarios

### Pure Logical Budget Test

```powershell
python -m experiments.scheduling.cache_budget_demo
```

Expected sequence: 2 MiB available with cache, task remains pending; 3 MiB after simulated cache clear, task starts and reserves all 3 MiB; completion restores 3 MiB. The final line should be `Cache budget test: PASS`.

This test does not start a worker process or allocate GPU memory.

### Extension Worker and Earlier Recovery Demos

```powershell
python -m experiments.scheduling.process_demo
python -m experiments.scheduling.crash_demo
python -m experiments.scheduling.retry_demo
```

`process_demo` and `retry_demo` execute the compiled vector-add/scale-add extension. `crash_demo` is the earlier controlled process-loss example.

### Real Qwen Worker, Cache, and Retry

```powershell
python -m experiments.scheduling.qwen_process_demo
python -m experiments.scheduling.qwen_cache_restart_demo
python -m experiments.scheduling.qwen_retry_demo
```

| Entry point | What to inspect |
| --- | --- |
| `qwen_process_demo` | Prefill miss, partial/exact reuse, reported cache bytes, updated logical budget |
| `qwen_cache_restart_demo` | Normal shutdown/restart; new process reports zero cache and misses on the same prompt |
| `qwen_retry_demo` | Intentional exit code 1, task requeue, successful attempt 2, stale result rejected, duplicate result without duplicate budget release |

The current retry task is prefill-only. A successful result does not mean it generated a natural-language response. Queue isolation in the restart scenarios is local to this test design.

### Admission and Request Batching

```powershell
python -m experiments.scheduling.qwen_real_admission_demo
python -m experiments.scheduling.qwen_batch_process_demo
python -m experiments.scheduling.qwen_batch_policy_benchmark
```

These use a real model process. A pending task is an expected outcome when admission rejects it. Batch policy timing measures request latency/throughput, not just one CUDA kernel. Current batching waits for a complete batch's generation call to finish; it is not token-level continuous batching.

## 6. Supplementary Gloo/DDP Exercises

```powershell
python -m experiments.distributed.gloo_all_reduce_demo
python -m experiments.distributed.gloo_gradient_sync_demo
python -m experiments.distributed.gloo_ddp_demo
```

These start two local CPU processes. Run one exercise at a time because the scripts use a shared rendezvous-file location. Example results include a reduced value of 3.0 and synchronized gradients of 5.0 with updated weights of 0.5.

## Validation Status of This Organization Pass

See [STRUCTURE_CHANGES.md](STRUCTURE_CHANGES.md) for the migration validation record. Historical performance results are not new measurements. Python import/build plumbing changed; existing function/class bodies and CUDA source were checked against the pre-migration snapshot.

Historical results remain in [BENCHMARKS.md](BENCHMARKS.md) and [PROGRESS.md](PROGRESS.md). See [ARCHITECTURE.md](ARCHITECTURE.md) for known cleanup work before a clean-install or public-release claim.

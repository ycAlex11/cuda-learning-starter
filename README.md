# CUDA and LLM Systems Learning Project

A hands-on learning project developed on a Windows laptop with a single GTX 1650 Max-Q GPU.

The project follows a practical path: write CUDA kernels, expose them to PyTorch, measure their behavior in model-shaped workloads, run real Qwen inference, and build a small task scheduler around persistent GPU worker processes.

This is a collection of experiments and reusable modules, not one application that must be run from beginning to end. It is a single-machine educational prototype, not a production inference server.

## Documentation

- [Code map and architecture](docs/ARCHITECTURE.md): implementation files, experiment entry points, and their relationships.
- [Running the experiments](docs/RUNNING.md): prerequisites, build commands, and a route from CUDA basics to worker recovery.
- [File migration map](docs/STRUCTURE_CHANGES.md): old paths, new paths, and validation notes.
- [Progress log](docs/PROGRESS.md): the chronological learning and validation record.
- [Benchmark results](docs/BENCHMARKS.md): historical measurements, input shapes, and observed limitations.

## Learning Path

| Stage | Work covered | Main locations |
| --- | --- | --- |
| CUDA fundamentals | Vector addition, indexing, device memory, correctness, CUDA Event timing | [vector_add](experiments/cuda_basics/vector_add/) |
| CUDA reduction | Shared-memory reduction, multi-pass reduction, two elements per thread, warp shuffle, block-size experiments | [reduction](experiments/cuda_basics/reduction/) |
| PyTorch CUDA extensions | Add-one, fused scale-add, Softmax, scaled Softmax, RMSNorm, fused residual + RMSNorm; FP32 and FP16 experiments | [cuda_extensions](cuda_extensions/), [operator experiments](experiments/cuda_operators/) |
| Model-shaped computation | Grouped-query attention, residual connections, a simplified decoder layer, profiling, gate/up projection fusion | [model_components](experiments/model_components/) |
| Real model inference | Qwen generation, prefill/decode, KV Cache, request batching, reusable prefix caches | [qwen_inference](experiments/qwen_inference/), [common](common/) |
| Scheduling and recovery | Logical budgets, CUDA memory reports, process workers, heartbeats, batching, retries, stale-result rejection, idempotent completion | [scheduler](scheduler/), [scheduling experiments](experiments/scheduling/) |
| Supplementary distributed exercises | CPU Gloo all-reduce, manual gradient synchronization, a two-process DDP example | [distributed](experiments/distributed/) |

Early implementations are retained to show the learning and optimization sequence. They are not all interchangeable implementations of the latest behavior.

## Important Distinctions

- **Qwen-shaped experiments** use synthetic inputs and simplified layers with Qwen-compatible dimensions. They are not measurements of the complete pretrained model.
- **Real Qwen experiments** load `Qwen/Qwen2.5-0.5B-Instruct` using Transformers.
- Custom CUDA operators are tested independently and in selected model-shaped experiments. The real Qwen worker does not automatically replace the model's operators with these extensions.
- Current request batching runs complete batches through `model.generate()`. It is not token-level continuous batching.
- Gloo/DDP exercises run on two local CPU processes. They do not demonstrate multi-machine or multi-GPU training.

## Repository Layout

```text
cuda_extensions/            CUDA kernels, wrappers, bindings, per-operator setup.py
  vector_add/
  softmax/
  rmsnorm/
common/                     Shared inference, cache, validation, timing and import helpers
scheduler/                  Task models, scheduling, metrics, batch formation, resource estimates
  workers/                  GPU execution and queue-driven worker loops
experiments/
  cuda_basics/
    vector_add/             Standalone vector-add starter and early experiments
    reduction/              Preserved reduction variants and PyTorch reference
    matmul_benchmark.py     Standalone PyTorch matrix-multiplication benchmark
  cuda_operators/           Custom operator correctness, timing and profiling
  model_components/         Synthetic attention, MLP and Qwen-shaped layers
  qwen_inference/           Real pretrained Qwen inference and cache experiments
  scheduling/               Scheduling, admission, batching, process and recovery demos
  distributed/             Local CPU Gloo/DDP exercises
scripts/                    Environment check and Windows builds
docs/                       Architecture, run guide, progress, benchmark records
build/                      Ignored local outputs; extensions load from build/python/
```

The virtual environment, compiled extensions, executable outputs, and Python bytecode are local artifacts, not the source implementation.

## Selected Historical Results

These are recorded observations on the development laptop, not results from a fresh benchmark run or general performance guarantees.

| Experiment | Observation |
| --- | --- |
| Reduction, 1,000,000 FP32 elements | Initial custom implementation: 0.1614 ms; final optimized custom implementation: 0.0527 ms; PyTorch: 0.0358 ms |
| Fused scale-add, 4,194,304 FP32 elements | PyTorch eager: 0.4769 ms and 32 MiB extra peak active memory; custom fused extension: 0.2392 ms and 16 MiB |
| Softmax, FP32 shape [4096, 256] | Custom baseline: 0.3379 ms; custom warp-shuffle: 0.1884 ms; PyTorch: 0.0614 ms |
| RMSNorm, FP32 shape [1, 256, 896] | Recorded PyTorch baseline: 0.0694 ms; custom extension: 0.0274 ms |
| Gate/up projection fusion | Preserved correctness but did not produce a stable speedup |
| Shared-prefix prefill, two requests | Full prefill: 13167.65 ms; shared-prefix path: 6845.17 ms, approximately 1.92x |

See [BENCHMARKS.md](docs/BENCHMARKS.md) for context. Operator-level speedups must not be presented as full-model generation speedups. Profiler tables identify hotspots; their timings are not substitutes for the standalone benchmark measurements.

## Scheduler and Cache Validation

The reusable Scheduler manages task state and resource reservations. Demo entry points create processes and move messages through queues; the Scheduler itself does not launch the model.

Validated scenarios include:

- Real GPU extension tasks and real Qwen prefill/generation tasks.
- Persistent workers processing successive requests and batches.
- Task and batch admission using logical budgets and reported CUDA free memory.
- Prefix matching, exact hits, KV-only entry repair, byte-budgeted LRU eviction.
- Cache-usage snapshots sent through worker heartbeats.
- Budget changes when cache usage grows or is cleared.
- Normal worker restart with an empty cache and independent communication queues.
- Intentional exit after computation but before reporting, followed by requeue and retry.
- Rejection of stale attempt results and idempotent handling of duplicate completions.

The latest Qwen retry demo exercises `qwen_prefill`; earlier progress entries also describe the previous generation-based experiment.

## Development Environment

- Windows / PowerShell.
- NVIDIA GeForce GTX 1650 Max-Q, 4 GiB VRAM; CUDA target `sm_75`.
- CUDA Toolkit 12.4 and Visual Studio 2022 Build Tools / MSVC x64.
- Python 3.11 virtual environment.
- PyTorch CUDA 12.4 build.

Build scripts currently assume specific Windows toolchain paths. Consult the [run guide](docs/RUNNING.md) before using a different machine. A clean-machine installation has not yet been validated.

## Start Here

From the project root, using the existing virtual environment:

```powershell
# Logical scheduler test: no model, CUDA extension, or GPU required
.\.venv\Scripts\python.exe -m experiments.scheduling.cache_budget_demo

# Check the local PyTorch/CUDA environment
.\.venv\Scripts\python.exe .\scripts\check_env.py
```

For the full progression, follow [Running the experiments](docs/RUNNING.md). Real-model experiments require model weights to be available locally or downloadable and may take substantially longer than the small scheduler tests.

## Limitations

- Single-machine processes and queues, not network RPC or a cluster scheduler.
- Logical reservations are accounting, not CUDA-enforced quotas or an OOM guarantee.
- Heartbeat reports are periodic and may be stale.
- Cache tensor payload bytes are not the same as allocator-reserved memory or peak device usage.
- Prefix caches are process-local, copied for request isolation, and not persisted across restarts.
- Cache integration currently covers the single-task prefill path, not batched generation.
- Attempt fencing protects task state and accounting; it does not guarantee exactly-once execution or exactly-once external side effects.
- Restart demos use separate queues; this is not a complete distributed worker-identity protocol.
- CUDA kernels have workload-specific shape and dtype constraints; there is no general autograd integration.
- Some experiment scripts still duplicate helpers or import helpers from other demos. These are identified in the code map and have not been silently refactored.

## Cleanup Status

Source files are now grouped by responsibility. Import paths, build scripts, and run commands were updated; kernel algorithms and existing Python function/class bodies were preserved (apart from a relocated-path error message). See [STRUCTURE_CHANGES.md](docs/STRUCTURE_CHANGES.md) for the migration map and validation scope. Historical benchmarks have not been replaced with new performance claims.

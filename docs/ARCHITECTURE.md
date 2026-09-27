# Code Map and Architecture

This document describes the reorganized source tree. Early learning variants are retained. See [STRUCTURE_CHANGES.md](STRUCTURE_CHANGES.md) for old-to-new paths; entry points run as modules from the project root.

## 1. Standalone CUDA: Learn the Execution Model

| File or directory | Role |
| --- | --- |
| [experiments/cuda_basics/vector_add/vector_add_starter.cu](../experiments/cuda_basics/vector_add/vector_add_starter.cu) | Standalone vector-add starter built by `build_cpp_cuda.ps1` |
| [experiments/cuda_basics/vector_add](../experiments/cuda_basics/vector_add/) | PyTorch vector-add timing and the native CUDA vector-add experiment |
| [experiments/cuda_basics/reduction/01_reduction.cu](../experiments/cuda_basics/reduction/01_reduction.cu) | Initial shared-memory, multi-pass reduction |
| [experiments/cuda_basics/reduction/02_reduction_two_elements.cu](../experiments/cuda_basics/reduction/02_reduction_two_elements.cu) | Two input elements per thread |
| [experiments/cuda_basics/reduction/03_reduction_warp_shuffle.cu](../experiments/cuda_basics/reduction/03_reduction_warp_shuffle.cu) | Warp-shuffle reduction variant |
| [experiments/cuda_basics/reduction/04_reduction_warp_shuffle_128.cu](../experiments/cuda_basics/reduction/04_reduction_warp_shuffle_128.cu) | 128-thread experiment |
| [experiments/cuda_basics/reduction/05_reduction_warp_shuffle_512.cu](../experiments/cuda_basics/reduction/05_reduction_warp_shuffle_512.cu) | 512-thread experiment |
| [experiments/cuda_basics/reduction/02_reduction.py](../experiments/cuda_basics/reduction/02_reduction.py) | PyTorch reduction reference |

These programs have their own entry points and do not depend on the Scheduler or Qwen. Keep the reduction variants as an explicit optimization history.

## 2. CUDA Extensions: Expose Kernels to Python

| CUDA implementation | Build definition | Python module | Main tests |
| --- | --- | --- | --- |
| [vector_add_cuda.cu](../cuda_extensions/vector_add/vector_add_cuda.cu) | [setup.py](../cuda_extensions/vector_add/setup.py) | `vector_add_cuda` | [vector_add_demo.py](../experiments/cuda_operators/vector_add_demo.py) |
| [softmax.cu](../cuda_extensions/softmax/softmax.cu) | [softmax/setup.py](../cuda_extensions/softmax/setup.py) | `softmax_cuda` | [softmax_test.py](../experiments/cuda_operators/softmax_test.py), [softmax_fp16_test.py](../experiments/cuda_operators/softmax_fp16_test.py), [softmax_scale_test.py](../experiments/cuda_operators/softmax_scale_test.py) |
| [rmsnorm.cu](../cuda_extensions/rmsnorm/rmsnorm.cu) | [rmsnorm/setup.py](../cuda_extensions/rmsnorm/setup.py) | `rmsnorm_cuda` | [rmsnorm_test.py](../experiments/cuda_operators/rmsnorm_test.py), [rmsnorm_profile.py](../experiments/cuda_operators/rmsnorm_profile.py) |

Each extension contains CUDA kernels, C++ tensor validation/launch wrappers, and Python bindings. The module name comes from the build definition; bound function names come from `m.def(...)`.

Windows scripts in [scripts](../scripts/) initialize the compiler environment, build from each operator's directory, and copy `.pyd` modules into `build/python/`. Consumers call `common.extension_loader.load_cuda_extension`, which initializes PyTorch and loads from that directory. Python module names and bound function names are unchanged.

The Softmax experiments are specialized around 256 columns. FP16 paths use FP32 intermediate arithmetic where implemented. These are forward-pass learning operators, not drop-in training replacements with custom backward implementations.

## 3. Model-Shaped Experiments vs. Real Qwen

Synthetic layers live in [experiments/model_components](../experiments/model_components/). Real pretrained-model experiments live in [experiments/qwen_inference](../experiments/qwen_inference/).

### Synthetic and Model-Shaped Experiments

| Files | Purpose |
| --- | --- |
| `attention_demo.py`, `attention_fp16_baseline.py`, `attention_fp16_extension.py` | Compare attention calculations and custom Softmax on synthetic tensors |
| `residual_demo.py` | Illustrate residual connections using a deliberately constructed toy network |
| `qwen_block_demo.py`, `qwen_block_benchmark.py`, `qwen_block_profile.py` | Build, time, and profile a simplified Qwen-shaped decoder layer |
| `mlp_fusion_benchmark.py` | Compare separate gate/up projections with a concatenated projection |

The Qwen-shaped layer uses random weights. It is not a complete implementation of the pretrained decoder, and timing it does not measure real-model generation throughput.

### Real-Model Experiments

| Files | Purpose |
| --- | --- |
| `qwen_inference.py` | Load the pretrained Qwen model and measure generation |
| `kv_cache_demo.py`, `prefill_decode_demo.py` | Inspect cache shapes and compare cached/uncached execution |
| `qwen_batch_demo.py`, `qwen_batch_benchmark.py` | Compare individual requests and batched generation |
| `prefix_cache_demo.py` | Validate and benchmark prefix reuse, exact hits, KV-only repair, and eviction |
| `prefix_cache_store_demo.py` | Small cache-lookup exercise without loading model weights |

Custom CUDA extensions are not automatically inserted into the real-model paths.

## 4. Shared Python Modules

| Module | Responsibility |
| --- | --- |
| [common/benchmarking.py](../common/benchmarking.py) | Warm-up and repeated CUDA Event timing; returns latency samples |
| [common/benchmark_reporting.py](../common/benchmark_reporting.py) | Earlier printed timing and peak-memory helpers; original defaults retained, no operator import required |
| [common/extension_loader.py](../common/extension_loader.py) | Import native modules from `build/python/` after initializing PyTorch |
| [common/model_loader.py](../common/model_loader.py) | Load a causal language model and tokenizer |
| [common/causal_lm.py](../common/causal_lm.py) | Prefill, decode step, greedy token selection, mask extension, suffix construction |
| [common/validation.py](../common/validation.py) | Compare tensors or final-position logits and next-token choices |
| [common/prefix_cache.py](../common/prefix_cache.py) | Cache entries, longest-prefix lookup, LRU eviction, byte accounting, cached request execution |

`PrefixCache` stores token-ID keys, KV tensors, and optional final-position logits. It does not own the model or choose which worker receives a request.

Exact hits with logits can return without a model call. An exact hit containing only KV data falls back to full prefill to repair the entry. Partial hits reuse an isolated copy of the cached prefix and evaluate the suffix.

Byte accounting measures tensor payloads. The byte limit controls retained entries after insertion; it does not cap peak allocation during model execution or cache copying.

## 5. Scheduler Implementation and Scenario Entry Points

### Reusable Implementation

| Module | Responsibility |
| --- | --- |
| [models.py](../scheduler/models.py) | `Task`, `TaskStatus`, `Worker`, and computed logical free budget |
| [core.py](../scheduler/core.py) | Registration, admission, task state, heartbeats, completion/failure handling, requeue |
| [metrics.py](../scheduler/metrics.py) | Scheduler counters |
| [batching.py](../scheduler/batching.py) | `TaskBatch`, FIFO batch formation, count/budget/wait constraints |
| [qwen_resources.py](../scheduler/qwen_resources.py) | KV-size estimate using fixed Qwen model dimensions |
| [gpu_worker.py](../scheduler/workers/gpu_worker.py) | Execute a CUDA extension task |
| [process_worker.py](../scheduler/workers/process_worker.py) | Queue-driven extension worker process loop |
| [qwen_process_worker.py](../scheduler/workers/qwen_process_worker.py) | Load Qwen, execute tasks/batches, own PrefixCache, send heartbeats/results |

`Worker` is a Scheduler-side record, not a process launcher. Demo `main()` functions create `multiprocessing.Process` objects explicitly.

### Scenario Entry Points

These scripts live in [experiments/scheduling](../experiments/scheduling/), separate from the reusable implementation.

| Group | Files |
| --- | --- |
| Logical scheduling and batching | `scheduling_demo.py`, `batch_demo.py`, `cache_budget_demo.py` |
| CUDA extension execution | `gpu_demo.py`, `process_demo.py`, `crash_demo.py`, `retry_demo.py` |
| Qwen resource admission | `qwen_resource_demo.py`, `qwen_admission_demo.py`, `qwen_real_admission_demo.py` |
| Real Qwen tasks and failure recovery | `qwen_process_demo.py`, `qwen_retry_demo.py` |
| Normal shutdown and cache restart | `qwen_cache_restart_demo.py` (normal shutdown, then restart) |
| Qwen request batching and policy measurements | `qwen_batch_process_demo.py`, `qwen_batch_policy_benchmark.py` |

These entry points mostly reuse the same Scheduler. A new demo does not mean a separate scheduler implementation.

### Runtime Ownership and Message Flow

1. The parent process creates the Scheduler, registers a Worker record, and submits Tasks.
2. The demo starts a child process running a worker loop.
3. The Scheduler checks capacity and marks a task as running; the demo places it on `task_queue`.
4. The child executes the operation and sends its result through `result_queue`.
5. The parent validates the reported attempt and updates task state and reservations.
6. A heartbeat thread inside the Qwen child independently sends resource snapshots through `heartbeat_queue`.

The cache is owned by the child process. Its task loop updates a scalar usage snapshot; the heartbeat thread reads that scalar rather than traversing a changing cache.

### Resource Accounting

- `total_vram_mb`: configured logical capacity.
- `used_vram_mb`: base/model reservation plus currently reserved tasks; not measured device usage.
- `reported_prefix_cache_bytes`: the latest accepted cache-usage snapshot.
- `free_vram_mb`: total minus used reservations minus reported cache bytes rounded up to MiB.
- `reported_free_vram_mb` / `reported_total_vram_mb`: separate device-level readings from CUDA.

Admission checks logical capacity and reported device free memory separately. Cache occupancy must not be deducted again from the device-free reading. Periodic reports and estimates do not provide hard memory isolation.

### Retry and Cache Boundaries

- A lost task is requeued in the Scheduler; it is not recovered from the old queue.
- A replacement process receives a newly scheduled attempt through fresh queues.
- The cache is not persisted across process restarts.
- Late attempt results are fenced, and duplicate completion reports do not release reservations twice.
- Batch completion currently reports whole-batch success/failure, not partial per-request success.
- The prefix-cache execution path is `qwen_prefill`; ordinary generation and batched generation do not use this custom cache.

## 6. Supplementary Distributed Exercises

[experiments/distributed](../experiments/distributed/) contains:

- `gloo_all_reduce_demo.py`: sum values across two local CPU ranks.
- `gloo_gradient_sync_demo.py`: manually average gradients before updating parameters.
- `gloo_ddp_demo.py`: demonstrate automatic gradient synchronization with DDP.

These small examples are independent of the task Scheduler and are not a multi-GPU training system.

## 7. Documentation and Local Artifacts

- [README](../README.md): project scope and navigation.
- [RUNNING](RUNNING.md): representative commands and prerequisites.
- [PROGRESS](PROGRESS.md): historical implementation and validation record.
- [BENCHMARKS](BENCHMARKS.md): historical measurements and their context.

`.venv/`, `build/`, `cuda_extensions/*/build/`, `.pyd` files, bytecode, and logs are generated/local artifacts. Do not mistake their presence for a portable source build or a verified clean installation.

## 8. Remaining Cleanup: Outside This Structural Migration

1. Consolidate reusable timing helpers without changing benchmark scope, warm-up, or synchronization behavior. Keep old implementations where they intentionally document learning steps.
2. The generic printed timing helper now lives in `common/benchmark_reporting.py`. Its unnecessary Softmax import was removed; further timing consolidation remains separate work.
3. Move shared process-start and heartbeat-consumption helpers out of scenario demos. For example, `qwen_real_admission_demo.py` imports a helper from `qwen_retry_demo.py`.
4. Standardize names such as `hearbeat_loop`, `exectue_qwen_task`, and `from_batch`, updating all callers together rather than renaming a single definition.
5. Align entry-point conventions and add explicit failure assertions where scripts currently only print correctness.
6. Capture install dependencies, validate a clean setup, and parameterize machine-specific build paths.
7. Before publishing, review benchmark comparability. In particular, FP16 residual-RMSNorm correctness uses an FP32-addition reference, while the current timing lambda uses eager FP16 addition; those are not identical numerical paths.

The migration changed file locations and import/build plumbing only. Existing function names, algorithms, scheduling policies, cache policies, and benchmark defaults were preserved. Cross-demo helper imports remain explicit at their new paths; no shared-runtime refactor was bundled into the move.

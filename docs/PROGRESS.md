# Project Progress

This is the chronological experiment log. Phase labels reflect the original learning sequence, and historical next-step notes are retained. For the current organization and runnable entry points, see [ARCHITECTURE.md](ARCHITECTURE.md) and [RUNNING.md](RUNNING.md).

## Phase 1 — CUDA Vector Add Benchmark

Completed:

- Implemented CUDA vector add.
- Added correctness checks, warm-up, CUDA Event timing, median/min/max latency, and effective bandwidth estimation.
- Compared a native CUDA implementation with a PyTorch baseline.

Observed result on 1,000,000 `float32` elements:

- Native CUDA vector add: about 0.084 ms.
- Effective bandwidth: about 143 GB/s.

## Phase 2 — CUDA Reduction Optimization

Completed:

- Implemented a baseline CUDA reduction.
- Used shared memory and multi-kernel reduction.
- Added two-elements-per-thread optimization.
- Added warp shuffle optimization.
- Compared results against PyTorch `torch.sum`.

Observed result on 1,000,000 `float32` values:

- Initial CUDA reduction: about 0.161 ms.
- Two-elements-per-thread reduction: about 0.090 ms.
- Warp-shuffle optimized reduction: about 0.053 ms.
- PyTorch sum baseline: about 0.036 ms.

Conclusion: the optimized custom reduction improved substantially over the initial implementation, but PyTorch remained faster due to its highly optimized kernels and runtime implementation.

## Phase 3 — PyTorch CUDA Extension

Completed:

- Built a C++/CUDA PyTorch extension targeting `sm_75`.
- Implemented `add_one`.
- Implemented fused `scale_add`: `output = input * scale + bias`.
- Added correctness checks and latency benchmarks.
- Compared fused execution with PyTorch eager `x * scale + bias`.
- Compared peak active memory usage.

Stable benchmark on 4,194,304 `float32` values:

- PyTorch eager `x * scale + bias`: about 0.477 ms.
- Fused CUDA extension: about 0.239 ms.
- PyTorch separate operations extra peak active memory: 32 MiB.
- Fused CUDA extension extra peak active memory: 16 MiB.

Conclusion: kernel fusion avoids an intermediate tensor and reduces both kernel launches and global-memory traffic.

## Phase 4 — Mini GPU Scheduler

Completed:

- Created generic `Task` and `Worker` data models.
- Added task states: `pending`, `running`, `succeeded`, and `failed`.
- Added logical VRAM-aware scheduling.
- Added worker heartbeat and timeout detection.
- Added worker-loss requeue logic.
- Added task retry attempts.
- Added stale completion rejection.
- Added idempotent duplicate completion handling.
- Added scheduler metrics.

Validated logical scheduler scenario:

- Submitted tasks: 4.
- Scheduled attempts: 5.
- Completed tasks: 3.
- Requeued tasks: 1.
- Rejected stale completions: 1.
- Duplicate completions: 1.

## Phase 5 — Process-Isolated GPU Worker and Failure Recovery

Status: completed as a single-machine prototype.

Implemented:

- A separate Python worker process using `multiprocessing.Process`.
- `task_queue` for scheduler-to-worker task delivery.
- `result_queue` for worker-to-scheduler completion reports.
- A structured worker result protocol:
  - `ok=True` with a result for successful execution.
  - `ok=False` with an error message for task execution failures.
- Worker-side execution of the real `scale_add` PyTorch CUDA extension.
- Scheduler-side handling for successful completion and reported task failure.
- A worker-crash simulation: no completion message, timeout, worker exit detection, logical VRAM release, and task requeue.
- A retry path: a replacement worker process receives the requeued task and successfully executes it on the GPU.
- Stale completion rejection using the task `attempt` number as a generation / fencing token.

Validated end-to-end retry scenario:

1. Attempt 1 was scheduled to `gpu-0`.
2. The first worker exited with code `1` without reporting completion.
3. The scheduler timed out, marked the worker as lost, released the 64 MiB logical reservation, and requeued `retry-001`.
4. A replacement `gpu-0` worker process sent a new heartbeat.
5. Attempt 2 executed the real CUDA extension successfully:
   `scale_add completed: num_elements=4194304, first_value=3.5, peak_extra_mib=32.00`.
6. A late completion from attempt 1 was rejected.
7. Final state: task succeeded, attempts = 2, worker free VRAM = 4096 MiB.

Final retry-demo metrics:

- Submitted tasks: 1.
- Scheduled attempts: 2.
- Completed tasks: 1.
- Requeued tasks: 1.
- Rejected completions: 1.

Key conclusion:

The prototype provides at-least-once task execution semantics. A task may be retried after worker loss, so completion reports must carry an attempt number. The scheduler accepts only the completion that matches the active attempt, preventing stale workers from overwriting newer task state.

## Qwen-Shaped Attention Experiment

Completed a standalone grouped-query attention experiment modeled after Qwen2.5-0.5B-Instruct.

- Implemented the attention path in PyTorch: `QK^T -> scale -> Softmax -> weights x V`.
- Used Qwen-shaped tensors: 14 Q heads, 2 KV heads, head dimension 64, sequence length 256.
- Integrated and verified the custom float32 CUDA warp-shuffle Softmax extension.
- Verified both Softmax output and full attention output against PyTorch.
- Benchmarked isolated Softmax and full attention.
- Measured FP16 versus FP32 latency and peak active memory.
- Observed that FP16 reduced memory use but was slower for full attention on GTX 1650 Max-Q, which has no Tensor Cores.

Next step: decide whether to generalize the CUDA Softmax extension for variable sequence lengths and FP16, or profile the current workload with Nsight tools.

## Qwen-Shaped Decoder Layer Profiling and MLP Fusion

Completed:

- Built a Qwen2.5-0.5B-shaped FP16 decoder-layer microbenchmark with:
  - hidden size 896
  - sequence length 256
  - 14 query heads, 2 KV heads, head dimension 64
  - MLP intermediate size 4864
- Implemented PyTorch grouped-query causal attention, SwiGLU MLP, RMSNorm, and residual connections.
- Integrated the custom FP16 CUDA RMSNorm extension into the complete layer.
- Verified PyTorch and custom-RMSNorm layer outputs:
  - Correctness: PASS
  - Maximum absolute error: 0.001953125
- Used `torch.profiler` with named regions to identify GPU hotspots.

Profiler conclusion:

- Full Qwen-shaped layer: 47.080 ms CUDA total.
- SwiGLU MLP: 39.616 ms, about 84.1% of layer time.
- QKV projections: 3.306 ms, about 7.0%.
- O projection plus residual: 2.177 ms, about 4.6%.
- Attention batched matrix multiplications: 1.247 ms, about 2.6%.
- Two RMSNorm calls: 0.293 ms, about 0.6%.

MLP fusion experiment:

- Combined `gate_proj` and `up_proj` weights into one `896 -> 9728` projection.
- Split the output back into two 4864-wide tensors before SwiGLU.
- Correctness: PASS, maximum absolute error 0.00000000.
- Stable measurements showed no speedup:
  - Separate gate + up MLP: about 46.33 ms.
  - Fused gate/up MLP: about 46.46 ms.

Key conclusion:

The MLP GEMMs, not RMSNorm or Softmax, dominate this Qwen-shaped workload. Concatenating gate and up projections does not reduce total GEMM work or output traffic, so it did not improve latency on the GTX 1650 Max-Q. This is a validated negative optimization result.

## Phase 5 — Real Qwen Worker Failure and Retry

Status: complete

Implemented and verified an end-to-end Qwen worker recovery experiment.

- Ran `Qwen/Qwen2.5-0.5B-Instruct` in a separate Python worker process.
- Started a heartbeat thread inside the worker process.
- Scheduler received worker heartbeats through `heartbeat_queue`.
- Submitted `qwen-retry-001` with an estimated 1 MiB KV Cache reservation.
- Attempt 1 executed Qwen inference, then intentionally exited before reporting completion.
- Scheduler received no completion result, detected the lost worker, released the task's logical KV Cache reservation, and requeued the task.
- Started a new worker process, which reloaded Qwen.
- Attempt 2 completed successfully and the Scheduler accepted the result.
- Verified stale completion fencing: a late completion from attempt 1 was rejected.
- Verified idempotent completion: a duplicate completion from attempt 2 was accepted without releasing resources twice.

Observed state transitions:

```text
pending
  -> running, attempt 1
  -> pending, worker lost and task requeued
  -> running, attempt 2
  -> succeeded
```

Observed resource-accounting transitions:

```text
Worker total VRAM: 4096 MiB
Model logical reservation: 1024 MiB
Free before task: 3072 MiB
Free after scheduling 1 MiB KV Cache: 3071 MiB
Free after worker loss and requeue: 3072 MiB
Free after retry completion: 3072 MiB
```

### Real CUDA VRAM Reporting and Admission Control

Added real CUDA VRAM reporting to Qwen worker heartbeats using:

```python
torch.cuda.mem_get_info()
```

Each heartbeat now reports:

- reported_free_vram_mb
- reported_total_vram_mb

The Scheduler stores these values separately from its logical used_vram_mb accounting.

Verified real admission control with a task requiring 2500 MiB:

```text
Scheduler logical free VRAM: 3072 MiB
Worker reported CUDA free VRAM: 2304 MiB
Task required VRAM: 2500 MiB
Scheduled task: None
Task status: pending
```

The task remained pending because logical capacity was sufficient but reported CUDA free VRAM was insufficient. This prevents sending an obviously non-admissible task to the Qwen worker.

## Request-Level Batching with a Real Qwen GPU Worker

Implemented request-level batching for real `Qwen/Qwen2.5-0.5B-Instruct` inference on the local GTX 1650 Max-Q. Each batch runs to completion before the next is dispatched; this is not token-level continuous batching.

Flow:

```text
pending Tasks
→ MicroBatcher forms a TaskBatch
→ Scheduler performs logical and reported-VRAM admission checks
→ TaskBatch is sent through a multiprocessing Queue
→ persistent Qwen worker performs batched model.generate()
→ Scheduler records completion and releases logical VRAM reservations
```

Validation:

- One persistent Qwen worker loaded the model once and processed multiple batches.
- Three requests with max_batch_size=2 became two batches:
  - inference-batch-001: two requests
  - inference-batch-002: one request
- All three tasks completed successfully in the normal path.
- Scheduler metrics after the normal run:
  - Submitted tasks: 3
  - Scheduled attempts: 3
  - Completed tasks: 3
  - Failed tasks: 0

Batch failure handling was also tested with a controlled worker-side failure:

- A failed batch released its complete logical VRAM reservation.
- Both tasks in the failed batch changed from running to failed.
- A later pending task still formed and completed in a new batch.
- Failure-test metrics:
  - Submitted tasks: 3
  - Scheduled attempts: 3
  - Completed tasks: 1
  - Failed tasks: 2

Current limitation: batch completion is atomic. The worker currently reports whole-batch success or failure; per-request partial completion is a future improvement.

## Batch Waiting Policy and End-to-End Latency

Added a time-based batching policy to `MicroBatcher`.

The batcher now dispatches a batch when either:

1. The number of pending tasks reaches `max_batch_size`; or
2. The oldest pending task has waited at least `max_batch_wait_ms`.

Added two scheduler-side latency measurements:

- `queue_wait_ms`: total time from task submission until batch dispatch.
- `intentional_batching_wait_ms`: time intentionally spent waiting for more requests after a worker became available.

Added `submitted_at` and `completed_at` fields to `Task`, enabling end-to-end task latency measurement:

```text
end_to_end_latency = completed_at - submitted_at
```

Created experiments/scheduling/qwen_batch_policy_benchmark.py, which runs the real Qwen worker under multiple batching policies and reports median task latency, median tail latency, and estimated request throughput.

## Prefix Cache Experiment Completed

Implemented and benchmarked shared-prefix KV Cache reuse with Qwen2.5-0.5B-Instruct. The experiment verifies token-level prefix compatibility, reuses a 951-token prefetched KV Cache across two requests, validates FP16 output agreement, and measures end-to-end prefill latency.

Result: shared-prefix cache reuse reduced median prefill latency from 13167.65 ms to 6845.17 ms, a 1.92x speedup.


## Prefix Cache: Worker Integration and Exact-Hit Fix

### Completed

- Created a PrefixCache inside the persistent Qwen worker process for reuse across tasks.
- Added the qwen_prefill task type to process inputs and store or reuse cache entries without generating a complete response.
- Cache entries store both KV tensors and logits for the final input position.
- On an exact hit with cached logits, return the cached result directly instead of passing an empty input to the model.
- On an exact hit without cached logits, recompute the full input and update the cache entry.
- Preserved partial-hit, cache-miss, and LRU eviction behavior.

### Validation Results

1. Cross-task cache reuse within a worker

- The first request missed the cache and stored an entry after processing.
- The second request reused an existing prefix and computed the additional input.
- The third request exactly matched a cached request and returned successfully.
- Final counters: cache_hits=2, cache_misses=1, cache_entries=2.
- All three tasks reached succeeded status.

2. Exact-hit output consistency

- Exact-match length: 965 tokens.
- The final-position logits returned from the cache exactly matched the previous computation.
- Correctness: True.
- Next token match: True.
- Maximum absolute error: 0.00000000.

3. KV-only cache entries without logits

- Test input: 4 tokens.
- The first call recomputed the input and stored the missing logits, returning reused tokens=0.
- The second call directly reused the complete cached result, returning reused tokens=4.
- Correctness: True.
- Maximum absolute error: 0.00000000.

### Current Scope

- The cache resides inside the worker process and does not persist after the process exits.
- Integration currently covers the single-task qwen_prefill path, not the batched generation path.
- Validation covers final-position logits and next-token selection, not complete generated sequences.
- No new performance measurements were added in this step; the results above validate functionality and correctness.

## Prefix Cache Byte Budget and LRU Eviction

Added a `max_bytes` limit to PrefixCache and measured the size of cached tensors:

- Calculated the tensor payload size of KV Cache and logits using `numel() * element_size()`.
- Test budget: `max_bytes=600000`.
- First request cache size: 414464 bytes, approximately 0.3953 MiB.
- Inserting the second request exceeded the budget and evicted the first entry.
- Recomputing and inserting the third request evicted the second entry.
- Final cache entry count: 1.
- Final cache size: 414464 bytes.
- LRU evictions: 2.

This demonstrates why cache capacity should account for KV tensor bytes rather than only the number of entries.

## Prefix Cache Usage Reporting and Scheduling Budget

Completed:

- The worker measures prefix-cache tensor bytes after task completion.
- The heartbeat thread reads the resource snapshot and periodically reports it to the Scheduler.
- The Scheduler records cache usage and deducts it when calculating the available logical budget.
- Cache bytes are rounded up when converted to MiB.

Real Qwen test:

- Cache usage: 878080 bytes.
- Available logical budget: updated from 3072 MiB to 3071 MiB.

Standalone scheduling test:

- Total budget: 4 MiB, with a base reservation of 1 MiB.
- With 1 MiB of cache usage, only 2 MiB remained, so a task requiring 3 MiB could not be scheduled.
- After simulating a cache clear, the available budget returned to 3 MiB and the task was scheduled successfully.
- Task completion released the task reservation, restoring the available budget to 3 MiB.
- All assertions passed.

Current limitations:

- Cache information comes from periodic heartbeats, not instantaneous measurements.
- Cached tensor bytes do not represent the entire process's actual GPU memory usage.
- Logical budget checks alone cannot guarantee that CUDA OOM will never occur.

## Prefix Cache Validation After Worker Restart

Test file: experiments/scheduling/qwen_cache_restart_demo.py

Validation procedure:

1. The first worker process performed prefill and created a cache of 414464 bytes.
2. The Scheduler recorded cache usage from a heartbeat, reducing the available logical budget to 3071 MiB.
3. The first process shut down normally, and a new process was started.
4. The new process used independent communication queues to avoid reading heartbeats left by the old process.
5. The new process reported cache usage of 0, restoring the Scheduler's available logical budget to 3072 MiB.
6. Submitting the same prompt to the new process produced hits=0 and misses=1.
7. The new process recomputed the input and built its own cache.

Conclusions:

- The cache is stored only in process memory and does not survive a process restart.
- The Scheduler retains the original Worker object and updates cache usage from new heartbeats.
- This test validates restart after normal shutdown, not recovery from an abnormal crash.

## Qwen Crash Recovery, Task Retry, and Cache Budget Updates

Test file: experiments/scheduling/qwen_retry_demo.py

Validation procedure:

1. The first qwen_prefill attempt established a prefix cache, then intentionally exited with code 1 before reporting its result.
2. The Scheduler returned the task to pending and released its reserved budget.
3. After confirming that the old process had exited, a replacement process was started with new communication queues.
4. The new process reported cache usage of 0.
5. The same task ran again with attempt=2 and completed successfully.
6. The retry returned matched_prefix_tokens=0, cache_hits=0, and cache_misses=1, confirming that it did not reuse the old process's cache.
7. The Scheduler received a subsequent heartbeat and recorded new cache usage of 414464 bytes.
8. The cache was charged as 1 MiB against the logical budget, leaving 3071 MiB available.

Stale and duplicate result checks:

- A late completion message for attempt=1 was rejected.
- A duplicate completion message for attempt=2 was recognized as already handled and returned True.
- After both checks, the task remained succeeded and the available budget remained 3071 MiB.
- The task reservation was not released twice.

Conclusions:

Task retry, cache usage reporting, and logical budget calculation are now integrated.
The new process builds its own cache, and the Scheduler updates its records using heartbeats from that process.

Validation boundaries:

- This test simulates an intentional exit after computation but before result reporting, not a power outage or network failure.
- Independent queues isolate messages from old and new processes, but do not constitute a complete production-grade process identity validation mechanism.
- Tasks may execute more than once. Preventing duplicate results from modifying the ledger twice does not imply exactly-once task execution.

## Source Layout Organization

The source tree is now organized by responsibility: CUDA basics, custom operator experiments, model-component experiments, real Qwen inference, scheduling experiments, and distributed exercises.

Scheduler implementation and worker loops are separate from their runnable demos. CUDA extension sources and build definitions are grouped by operator. Imports, build output paths, editor search paths and documentation commands were updated; algorithms, scheduling/cache policies and benchmark defaults were preserved.

Validation covered Python syntax and existing function-body comparisons, logical scheduler scenarios, extension loading and small GPU correctness checks, a real Qwen cache workflow and crash/retry workflow, a Qwen-shaped decoder example, and CPU DDP. A fresh Softmax rebuild remains unverified because the nested nvcc invocation could not locate cl.exe; existing binaries were retained and tested.

See [STRUCTURE_CHANGES.md](STRUCTURE_CHANGES.md) for the complete old-to-new file map and validation limits, and [RUNNING.md](RUNNING.md) for the new commands. No historical performance results were replaced by this migration.

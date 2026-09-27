# Source Layout Migration

## Scope

The project was reorganized by responsibility. This was a structural migration, not an algorithm or feature refactor.

- Retained every original Python and CUDA source file at its mapped location.
- Preserved existing function/class bodies, kernel calculations, resource accounting, cache policies, retry handling, benchmark scopes and defaults. The standalone matmul benchmark's environment-check error message now points to its new path.
- Updated imports, package entry points, build paths, editor search paths, documentation links and run commands.
- Removed the unused Softmax import from the generic printed benchmark helper. It now lives in `common/benchmark_reporting.py`; its timing and memory-measurement implementations did not change.
- Added `common/extension_loader.py` solely to initialize PyTorch and locate compiled modules under `build/python/`.
- Did not rename public operator functions or Python extension modules. For example, `softmax_cuda.softmax_forward_warp` is unchanged.
- Did not fix existing demo semantics or printed labels as part of the move. For example, `process_demo.py` currently submits operation `sth` to exercise the failure path.

## Running After the Move

From the project root with the virtual environment activated:

```powershell
python -m experiments.scheduling.cache_budget_demo
python -m experiments.scheduling.qwen_process_demo
python -m experiments.cuda_operators.softmax_test
python -m experiments.model_components.qwen_block_demo
```

Do not use old commands such as `python -m scheduler.qwen_process_demo`. See [RUNNING.md](RUNNING.md) for the updated commands. Run experiment modules from the project root so `common`, `scheduler`, and `experiments` resolve consistently.

## Old-to-New File Map

| Original source path | Current source path |
| --- | --- |
| `cpp_cuda/vector_add.cu` | [experiments/cuda_basics/vector_add/vector_add_starter.cu](../experiments/cuda_basics/vector_add/vector_add_starter.cu) |
| `experiments/phase1/01_torch_vector_add.py` | [experiments/cuda_basics/vector_add/01_torch_vector_add.py](../experiments/cuda_basics/vector_add/01_torch_vector_add.py) |
| `experiments/phase1/02_vector_add.cu` | [experiments/cuda_basics/vector_add/02_vector_add.cu](../experiments/cuda_basics/vector_add/02_vector_add.cu) |
| `experiments/phase2/01_reduction.cu` | [experiments/cuda_basics/reduction/01_reduction.cu](../experiments/cuda_basics/reduction/01_reduction.cu) |
| `experiments/phase2/02_reduction_two_elements.cu` | [experiments/cuda_basics/reduction/02_reduction_two_elements.cu](../experiments/cuda_basics/reduction/02_reduction_two_elements.cu) |
| `experiments/phase2/02_reduction.py` | [experiments/cuda_basics/reduction/02_reduction.py](../experiments/cuda_basics/reduction/02_reduction.py) |
| `experiments/phase2/03_reduction_warp_shuffle.cu` | [experiments/cuda_basics/reduction/03_reduction_warp_shuffle.cu](../experiments/cuda_basics/reduction/03_reduction_warp_shuffle.cu) |
| `experiments/phase2/04_reduction_warp_shuffle_128.cu` | [experiments/cuda_basics/reduction/04_reduction_warp_shuffle_128.cu](../experiments/cuda_basics/reduction/04_reduction_warp_shuffle_128.cu) |
| `experiments/phase2/05_reduction_warp_shuffle_512.cu` | [experiments/cuda_basics/reduction/05_reduction_warp_shuffle_512.cu](../experiments/cuda_basics/reduction/05_reduction_warp_shuffle_512.cu) |
| `experiments/qwen_inference/attention_demo.py` | [experiments/model_components/attention_demo.py](../experiments/model_components/attention_demo.py) |
| `experiments/qwen_inference/attention_fp16_baseline.py` | [experiments/model_components/attention_fp16_baseline.py](../experiments/model_components/attention_fp16_baseline.py) |
| `experiments/qwen_inference/attention_fp16_extension.py` | [experiments/model_components/attention_fp16_extension.py](../experiments/model_components/attention_fp16_extension.py) |
| `experiments/qwen_inference/mlp_fusion_benchmark.py` | [experiments/model_components/mlp_fusion_benchmark.py](../experiments/model_components/mlp_fusion_benchmark.py) |
| `experiments/qwen_inference/qwen_block_benchmark.py` | [experiments/model_components/qwen_block_benchmark.py](../experiments/model_components/qwen_block_benchmark.py) |
| `experiments/qwen_inference/qwen_block_demo.py` | [experiments/model_components/qwen_block_demo.py](../experiments/model_components/qwen_block_demo.py) |
| `experiments/qwen_inference/qwen_block_profile.py` | [experiments/model_components/qwen_block_profile.py](../experiments/model_components/qwen_block_profile.py) |
| `experiments/qwen_inference/residual_demo.py` | [experiments/model_components/residual_demo.py](../experiments/model_components/residual_demo.py) |
| `extension/rmsnorm.cu` | [cuda_extensions/rmsnorm/rmsnorm.cu](../cuda_extensions/rmsnorm/rmsnorm.cu) |
| `extension/setup_rmsnorm.py` | [cuda_extensions/rmsnorm/setup.py](../cuda_extensions/rmsnorm/setup.py) |
| `extension/setup_softmax.py` | [cuda_extensions/softmax/setup.py](../cuda_extensions/softmax/setup.py) |
| `extension/setup.py` | [cuda_extensions/vector_add/setup.py](../cuda_extensions/vector_add/setup.py) |
| `extension/softmax.cu` | [cuda_extensions/softmax/softmax.cu](../cuda_extensions/softmax/softmax.cu) |
| `extension/vector_add_cuda.cu` | [cuda_extensions/vector_add/vector_add_cuda.cu](../cuda_extensions/vector_add/vector_add_cuda.cu) |
| `python/benchmark_softmax.py` | [common/benchmark_reporting.py](../common/benchmark_reporting.py) |
| `python/benchmark.py` | [experiments/cuda_basics/matmul_benchmark.py](../experiments/cuda_basics/matmul_benchmark.py) |
| `python/check_env.py` | [scripts/check_env.py](../scripts/check_env.py) |
| `python/extension_demo.py` | [experiments/cuda_operators/vector_add_demo.py](../experiments/cuda_operators/vector_add_demo.py) |
| `python/rmsnorm_profile.py` | [experiments/cuda_operators/rmsnorm_profile.py](../experiments/cuda_operators/rmsnorm_profile.py) |
| `python/rmsnorm_test.py` | [experiments/cuda_operators/rmsnorm_test.py](../experiments/cuda_operators/rmsnorm_test.py) |
| `python/softmax_fp16_test.py` | [experiments/cuda_operators/softmax_fp16_test.py](../experiments/cuda_operators/softmax_fp16_test.py) |
| `python/softmax_scale_test.py` | [experiments/cuda_operators/softmax_scale_test.py](../experiments/cuda_operators/softmax_scale_test.py) |
| `python/softmax_test.py` | [experiments/cuda_operators/softmax_test.py](../experiments/cuda_operators/softmax_test.py) |
| `scheduler/batch_demo.py` | [experiments/scheduling/batch_demo.py](../experiments/scheduling/batch_demo.py) |
| `scheduler/cache_budget_demo.py` | [experiments/scheduling/cache_budget_demo.py](../experiments/scheduling/cache_budget_demo.py) |
| `scheduler/crash_demo.py` | [experiments/scheduling/crash_demo.py](../experiments/scheduling/crash_demo.py) |
| `scheduler/demo.py` | [experiments/scheduling/scheduling_demo.py](../experiments/scheduling/scheduling_demo.py) |
| `scheduler/gpu_demo.py` | [experiments/scheduling/gpu_demo.py](../experiments/scheduling/gpu_demo.py) |
| `scheduler/gpu_worker.py` | [scheduler/workers/gpu_worker.py](../scheduler/workers/gpu_worker.py) |
| `scheduler/process_demo.py` | [experiments/scheduling/process_demo.py](../experiments/scheduling/process_demo.py) |
| `scheduler/process_worker.py` | [scheduler/workers/process_worker.py](../scheduler/workers/process_worker.py) |
| `scheduler/qwen_admission_demo.py` | [experiments/scheduling/qwen_admission_demo.py](../experiments/scheduling/qwen_admission_demo.py) |
| `scheduler/qwen_batch_policy_benchmark.py` | [experiments/scheduling/qwen_batch_policy_benchmark.py](../experiments/scheduling/qwen_batch_policy_benchmark.py) |
| `scheduler/qwen_batch_process_demo.py` | [experiments/scheduling/qwen_batch_process_demo.py](../experiments/scheduling/qwen_batch_process_demo.py) |
| `scheduler/qwen_crash_restart.py` | [experiments/scheduling/qwen_cache_restart_demo.py](../experiments/scheduling/qwen_cache_restart_demo.py) |
| `scheduler/qwen_process_demo.py` | [experiments/scheduling/qwen_process_demo.py](../experiments/scheduling/qwen_process_demo.py) |
| `scheduler/qwen_process_worker.py` | [scheduler/workers/qwen_process_worker.py](../scheduler/workers/qwen_process_worker.py) |
| `scheduler/qwen_real_admission_demo.py` | [experiments/scheduling/qwen_real_admission_demo.py](../experiments/scheduling/qwen_real_admission_demo.py) |
| `scheduler/qwen_resource_demo.py` | [experiments/scheduling/qwen_resource_demo.py](../experiments/scheduling/qwen_resource_demo.py) |
| `scheduler/qwen_retry_demo.py` | [experiments/scheduling/qwen_retry_demo.py](../experiments/scheduling/qwen_retry_demo.py) |
| `scheduler/retry_demo.py` | [experiments/scheduling/retry_demo.py](../experiments/scheduling/retry_demo.py) |

The old `python/benchmark.py` is an executable matrix-multiplication experiment, not a generic benchmark utility; it therefore moved to `experiments/cuda_basics/matmul_benchmark.py`.

## Generated Artifacts and Recovery

- Active native modules now load from `build/python/`.
- The three pre-existing `.pyd` files were copied there and their hashes verified against the originals.
- Original leftover build outputs, compiled modules and bytecode were preserved under `build/legacy_artifacts/`, not deleted.
- An empty directory mistakenly named `common/__init__.py` was moved to `build/legacy_artifacts/common-init-directory/` and replaced with a Python package file.
- A pre-migration text snapshot is available locally at `build/layout_migration/before.json`. It is ignored build data, not a runtime dependency.
- Keep `.venv/`, build outputs and machine-specific artifacts out of a future source publication.

## Validation

### Passed

- Parsed/compiled the current Python source files without running their main functions.
- Compared all 71 pre-existing Python/CUDA source files with their mapped files: original Python function/class bodies and CUDA source match, allowing a final newline and the one relocated-path error message.
- Parsed the PowerShell build scripts.
- Ran logical scheduling, cache-budget and batching demos from their new module paths.
- Loaded all three original compiled extensions from `build/python/`.
- Ran small GPU correctness checks for scale-add, FP32/FP16 Softmax and FP32/FP16 RMSNorm.
- Ran the extension process demo: its intentional unsupported-operation failure was reported, and its logical resource budget was restored.
- Ran the real Qwen process demo in offline mode using the existing local model cache. All three prefill requests succeeded, ending with two cache hits, one miss, two entries, 878080 reported bytes and 3071 MiB logical free budget.

- Ran the real Qwen retry demo in offline mode: attempt one exited, attempt two succeeded, reported cache usage was 414464 bytes, stale completion was rejected, and duplicate completion did not change the 3071 MiB budget.
- Ran the relocated Qwen-shaped decoder demo and the prefix-cache store demo.
- Ran the local two-process CPU DDP demo: both ranks obtained gradient 5.0 and updated weight 0.5.
- Checked local project imports, documentation links, fenced blocks, and documented module entry points.

### Not Fully Verified

- A fresh Softmax rebuild was attempted. The build found the relocated source and emitted its build files, but `nvcc` failed with `Cannot find compiler 'cl.exe' in PATH`. A separate VS developer-shell check located and invoked MSVC; this does not establish that the nested extension build environment is correct.
- Existing native binaries remain usable; they are not proof of a successful rebuild from the new layout. The other native builds were not rerun in this pass.
- The full historical GPU benchmark suite and clean-machine installation were not rerun. Existing benchmark numbers remain historical measurements.

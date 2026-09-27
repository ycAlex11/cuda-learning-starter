#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
  cudaError_t status = (call); \
  if (status != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d CUDA error: %s\\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) c[index] = a[index] + b[index];
}

int main() {
  constexpr int n = 1 << 24;  // Three buffers use about 192 MiB.
  constexpr int threads = 256;
  constexpr int iterations = 100;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  float *a, *b, *c;
  CUDA_CHECK(cudaMalloc(&a, bytes));
  CUDA_CHECK(cudaMalloc(&b, bytes));
  CUDA_CHECK(cudaMalloc(&c, bytes));
  CUDA_CHECK(cudaMemset(a, 0, bytes));
  CUDA_CHECK(cudaMemset(b, 0, bytes));
  const int blocks = (n + threads - 1) / threads;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i) vector_add<<<blocks, threads>>>(a, b, c, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  float first_value = -1.0f;
  CUDA_CHECK(cudaMemcpy(&first_value, c, sizeof(float), cudaMemcpyDeviceToHost));
  const double bandwidth_gbs = (3.0 * bytes * iterations) / (elapsed_ms * 1e-3) / 1e9;
  std::printf("vector_add: %d values, %.3f ms/iteration, %.2f GB/s, c[0]=%.1f\\n", n,
              elapsed_ms / iterations, bandwidth_gbs, first_value);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(c));
  return 0;
}

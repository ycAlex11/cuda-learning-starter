#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
__global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    int n 
){

    int index = blockIdx.x *blockDim.x +threadIdx.x;

    if (index<n){
        c[index] = a[index] + b[index];        
    }
}

int main(){

    constexpr int n = 1000000;
    const size_t  bytes = n*sizeof(float);

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 10);
    }
    float* d_a;
    float* d_b;
    float* d_c;

    cudaMalloc(&d_a,bytes);
    cudaMalloc(&d_b,bytes);
    cudaMalloc(&d_c,bytes);
    cudaMemcpy(d_a,h_a.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b.data(),bytes,cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n+threads-1)/threads;
    std::printf("Launch: %d blocks x %d threads\n", blocks, threads);
    vector_add<<<blocks,threads>>>(d_a,d_b,d_c,n);

    cudaError_t status = cudaGetLastError();

    if (status != cudaSuccess) {
        std::fprintf(stderr, "Kernel launch failed: %s\n",
                    cudaGetErrorString(status));
        return 1;
    }
    cudaMemcpy(h_c.data(),d_c,bytes,cudaMemcpyDeviceToHost);
    bool correct = true;

    for (int i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];

        if (std::fabs(h_c[i] - expected) > 1e-5f) {
            std::printf("Mismatch at index %d: expected %.1f, got %.1f\n",
                        i, expected, h_c[i]);
            correct = false;
            break;
        }
    }

    if (!correct) {
        return 1;
    }

    std::printf("Correctness: PASS (%d elements)\n", n);
    /*
    for (int i = 0; i < n; ++i) {
        std::printf("c[%d] = %.1f\n", i, h_c[i]);
    }
    std::printf("c[0] = %.1f\n", h_c[0]);
    std::printf("c[n / 2] = %.1f\n", h_c[n / 2]);
    std::printf("c[n - 1] = %.1f\n", h_c[n - 1]);
    */
    constexpr int warmup = 10;

    for (int i = 0; i < warmup; ++i) {
        vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    }

    status = cudaDeviceSynchronize();

    if (status != cudaSuccess) {
        std::fprintf(stderr, "Warm-up failed: %s\n",
                    cudaGetErrorString(status));
        return 1;
    }
    
    constexpr int samples = 100;
std::vector<float> times_ms;
times_ms.reserve(samples);

cudaEvent_t start;
cudaEvent_t end;

cudaEventCreate(&start);
cudaEventCreate(&end);

for (int i = 0; i < samples; ++i) {
    cudaEventRecord(start);

    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, end);
    times_ms.push_back(elapsed_ms);
}

std::sort(times_ms.begin(), times_ms.end());

float median_ms = (times_ms[49] + times_ms[50]) / 2.0f;

std::printf("CUDA kernel median: %.4f ms\n", median_ms);
std::printf("CUDA kernel min / max: %.4f / %.4f ms\n",
            times_ms.front(), times_ms.back());

double bandwidth_gbs = (3.0 * bytes) / (median_ms * 1e-3) / 1e9;

std::printf("Estimated effective bandwidth: %.2f GB/s\n", bandwidth_gbs);

cudaEventDestroy(start);
cudaEventDestroy(end);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;

}

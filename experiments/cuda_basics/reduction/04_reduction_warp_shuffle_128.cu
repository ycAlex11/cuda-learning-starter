#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
__global__ void block_sum(const float* input, float* partial_sums,int n)
{
    __shared__ float shared[128];
    int tid = threadIdx.x;
    int first_index = blockIdx.x * blockDim.x * 2 + tid;
    int second_index = first_index + blockDim.x;
    float local_sum = 0.0f;

    if (first_index < n) {
        local_sum += input[first_index];
    }

    if (second_index < n) {
        local_sum += input[second_index];
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride /= 2) {
            if (tid < stride) {
                shared[tid] += shared[tid + stride];
            }

            __syncthreads();
        }

    if (tid < 32) {
        float value = shared[tid] + shared[tid + 32];

        for (int offset = 16; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }

        if (tid == 0) {
            partial_sums[blockIdx.x] = value;
        }
    }

}

float* reduce_gpu(
    float* d_input,
    float* d_partials,
    float* d_next,
    int n,
    int threads
) {
    float* current_input = d_input;
    float* current_output = d_partials;
    int current_n = n;

    while (current_n > 1) {
        int elements_per_block = threads * 2;

        int current_blocks = (
            current_n + elements_per_block - 1
        ) / elements_per_block;

        block_sum<<<current_blocks, threads>>>(
            current_input, current_output, current_n
        );

        cudaError_t status = cudaGetLastError();
        if (status != cudaSuccess) {
            std::fprintf(stderr, "Reduction kernel launch failed: %s\n",
                         cudaGetErrorString(status));
            return nullptr;
        }

        current_n = current_blocks;

        float* previous_output = current_output;

        if (current_output == d_partials) {
            current_output = d_next;
        } else {
            current_output = d_partials;
        }

        current_input = previous_output;
    }

    return current_input;
}

int main(){

    constexpr int threads = 128;
    constexpr int n =1'000'000;
    constexpr int elements_per_block = threads * 2;
constexpr int blocks = (n + elements_per_block - 1) / elements_per_block;
    const size_t input_bytes = n* sizeof(float);
    const size_t partial_bytes = blocks * sizeof(float);

    std::vector<float> h_input(n);
    

    float cpu_sum = 0.0f;
    for(int i =0 ;i<n;++i){
        h_input[i] = 1.0f;
        cpu_sum+=h_input[i];
    }
    float* d_input;
    float* d_partials;
    float* d_next;
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_partials, partial_bytes);
    cudaMalloc(&d_next, partial_bytes);

    cudaMemcpy(d_input, h_input.data(), input_bytes,
               cudaMemcpyHostToDevice);
    //block_sum<<<blocks,threads>>>(d_input, d_partials,n);
    float* final_ptr = reduce_gpu(
    d_input, d_partials, d_next, n, threads
    );

    

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Kernel launch failed: %s\n",
                     cudaGetErrorString(status));
        return 1;
    }
    //final_sum<<<1, threads>>>(d_partials, d_final, blocks);

status = cudaGetLastError();
if (status != cudaSuccess) {
    std::fprintf(stderr, "Final kernel launch failed: %s\n",
                 cudaGetErrorString(status));
    return 1;
}
    float gpu_sum = 0.0f;

    //cudaMemcpy(&gpu_sum, d_partials, sizeof(float),cudaMemcpyDeviceToHost);
    
    //cudaMemcpy(h_partials.data(), d_partials, partial_bytes,cudaMemcpyDeviceToHost);
    
    /*
    for (int block = 0; block < blocks; ++block) {
        std::printf("partial[%d] = %.1f\n", block, h_partials[block]);
        gpu_sum += h_partials[block];
    }*/
    cudaMemcpy(&gpu_sum, final_ptr, sizeof(float),
           cudaMemcpyDeviceToHost);
    std::printf("CPU sum: %.1f\n", cpu_sum);
    std::printf("GPU sum: %.1f\n", gpu_sum);
    if (std::fabs(cpu_sum - gpu_sum) > 1e-5f) {
        std::fprintf(stderr, "Correctness: FAIL\n");
        return 1;
    }

    std::printf("Correctness: PASS\n");
    constexpr int warmup = 10;

for (int i = 0; i < warmup; ++i) {
    float* warmup_result = reduce_gpu(
        d_input, d_partials, d_next, n, threads
    );

    if (warmup_result == nullptr) {
        return 1;
    }
}

cudaDeviceSynchronize();

constexpr int samples = 100;

std::vector<float> times;
times.reserve(samples);

cudaEvent_t start;
cudaEvent_t end;

cudaEventCreate(&start);
cudaEventCreate(&end);

for (int i = 0; i < samples; ++i) {
    cudaEventRecord(start);

    final_ptr = reduce_gpu(
        d_input, d_partials, d_next, n, threads
    );

    if (final_ptr == nullptr) {
        return 1;
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, end);

    times.push_back(elapsed_ms);
}

cudaEventDestroy(start);
cudaEventDestroy(end);

std::sort(times.begin(), times.end());

float median_ms = 0.5f * (
    times[samples / 2 - 1] + times[samples / 2]
);

std::printf("CUDA reduction median: %.4f ms\n", median_ms);
std::printf("CUDA reduction min / max: %.4f / %.4f ms\n",
            times.front(), times.back());

    cudaFree(d_input);
    cudaFree(d_partials);
    cudaFree(d_next);
    return 0;
}


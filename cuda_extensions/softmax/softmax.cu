
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>


__global__ void softmax_forward_warp_kernel(const float* input,float* output,int cols){
    const int row = blockIdx.x;
    const int col = threadIdx.x;
    const int index = row*cols +col;
    __shared__ float shared[256];
    shared[col] = (col < cols) ? input[index] : -INFINITY;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride >= 64; stride /= 2) {
        if (col < stride) {
            shared[col] = fmaxf(shared[col], shared[col + stride]);
        }
        __syncthreads();
    }

    if(col<32){
        float value = fmaxf(shared[col], shared[col + 32]);
        for (int offset = 16; offset > 0; offset /= 2) {
            value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
        }
        if (col == 0) {
        shared[0] = value;
        }

    }
    __syncthreads();
    const float row_max = shared[0];
    __syncthreads();


    const float exp_value =(col < cols) ? expf(input[index] - row_max) : 0.0f;
    shared[col] = exp_value;
    __syncthreads();
    for (int stride = blockDim.x/2; stride >= 64;stride/=2){
        if(col<stride){
            shared[col] = shared[col] +shared[col+stride];
        }
        __syncthreads();
    }

    if(col<32){

        float sum_value = shared[col] + shared[col + 32];
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_value += __shfl_down_sync(
            0xffffffff, sum_value, offset
        );
        }
        if (col == 0) {
            shared[0] = sum_value;
        }

    }
    __syncthreads();

    const float row_sum = shared[0];
    if (col < cols) {
    output[index] = exp_value / row_sum;
    }
    

}


__global__ void softmax_forward_warp_scale_kernel(const float* input,float* output,int cols,float scale){
    const int row = blockIdx.x;
    const int col = threadIdx.x;
    const int index = row*cols +col;
    __shared__ float shared[256];
    shared[col] = (col < cols) ? input[index]*scale : -INFINITY;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride >= 64; stride /= 2) {
        if (col < stride) {
            shared[col] = fmaxf(shared[col], shared[col + stride]);
        }
        __syncthreads();
    }

    if(col<32){
        float value = fmaxf(shared[col], shared[col + 32]);
        for (int offset = 16; offset > 0; offset /= 2) {
            value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
        }
        if (col == 0) {
        shared[0] = value;
        }

    }
    __syncthreads();
    const float row_max = shared[0];
    __syncthreads();


    const float exp_value =(col < cols) ? expf(input[index]*scale - row_max) : 0.0f;
    shared[col] = exp_value;
    __syncthreads();
    for (int stride = blockDim.x/2; stride >= 64;stride/=2){
        if(col<stride){
            shared[col] = shared[col] +shared[col+stride];
        }
        __syncthreads();
    }

    if(col<32){

        float sum_value = shared[col] + shared[col + 32];
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_value += __shfl_down_sync(
            0xffffffff, sum_value, offset
        );
        }
        if (col == 0) {
            shared[0] = sum_value;
        }

    }
    __syncthreads();

    const float row_sum = shared[0];
    if (col < cols) {
    output[index] = exp_value / row_sum;
    }
    

}


__global__ void softmax_forward_warp_half_scale_kernel(
    const __half* input,
    __half* output,
    int cols,
    float scale
) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;
    const int index = row*cols +col;
    __shared__ float shared[256];
    shared[col] = (col < cols) ? __half2float(input[index])*scale : -INFINITY;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride >= 64; stride /= 2) {
        if (col < stride) {
            shared[col] = fmaxf(shared[col], shared[col + stride]);
        }
        __syncthreads();
    }

    if(col<32){
        float value = fmaxf(shared[col], shared[col + 32]);
        for (int offset = 16; offset > 0; offset /= 2) {
            value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
        }
        if (col == 0) {
        shared[0] = value;
        }

    }
    __syncthreads();
    const float row_max = shared[0];
    __syncthreads();


    const float exp_value =(col < cols) ? expf(__half2float(input[index]) * scale - row_max) : 0.0f;
    shared[col] = exp_value;
    __syncthreads();
    for (int stride = blockDim.x/2; stride >= 64;stride/=2){
        if(col<stride){
            shared[col] = shared[col] +shared[col+stride];
        }
        __syncthreads();
    }

    if(col<32){

        float sum_value = shared[col] + shared[col + 32];
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_value += __shfl_down_sync(
            0xffffffff, sum_value, offset
        );
        }
        if (col == 0) {
            shared[0] = sum_value;
        }

    }
    __syncthreads();

    const float row_sum = shared[0];
    if (col < cols) {
    output[index] = __float2half(exp_value / row_sum);
    }
    

}



__global__ void softmax_forward_warp_half_kernel(
    const __half* input,
    __half* output,
    int cols
) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;
    const int index = row*cols +col;
    __shared__ float shared[256];
    shared[col] = (col < cols) ? __half2float(input[index]) : -INFINITY;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride >= 64; stride /= 2) {
        if (col < stride) {
            shared[col] = fmaxf(shared[col], shared[col + stride]);
        }
        __syncthreads();
    }

    if(col<32){
        float value = fmaxf(shared[col], shared[col + 32]);
        for (int offset = 16; offset > 0; offset /= 2) {
            value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
        }
        if (col == 0) {
        shared[0] = value;
        }

    }
    __syncthreads();
    const float row_max = shared[0];
    __syncthreads();


    const float exp_value =(col < cols) ? expf(__half2float(input[index]) - row_max) : 0.0f;
    shared[col] = exp_value;
    __syncthreads();
    for (int stride = blockDim.x/2; stride >= 64;stride/=2){
        if(col<stride){
            shared[col] = shared[col] +shared[col+stride];
        }
        __syncthreads();
    }

    if(col<32){

        float sum_value = shared[col] + shared[col + 32];
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_value += __shfl_down_sync(
            0xffffffff, sum_value, offset
        );
        }
        if (col == 0) {
            shared[0] = sum_value;
        }

    }
    __syncthreads();

    const float row_sum = shared[0];
    if (col < cols) {
    output[index] = __float2half(exp_value / row_sum);
    }
    

}

__global__ void softmax_forward_kernel(const float* input, float* output,int cols){
    const int row = blockIdx.x;
    const int col = threadIdx.x;
    const int index = row*cols +col;

    __shared__ float shared[256];
    shared[col] = (col < cols) ? input[index] : -INFINITY;
    __syncthreads();
    for (int stride = blockDim.x/2; stride>0;stride/=2){
        if(col<stride){
            shared[col] = fmax(shared[col],shared[col+stride]);
        }

        __syncthreads();
    }
    const float row_max = shared[0];
    __syncthreads();
    const float exp_value =(col < cols) ? expf(input[index] - row_max) : 0.0f;

    shared[col] = exp_value;
    __syncthreads();
    for (int stride = blockDim.x/2; stride>0;stride/=2){
        if(col<stride){
            shared[col] = shared[col] +shared[col+stride];
        }
        __syncthreads();
    }
    const float row_sum = shared[0];

    if (col<cols){
        output[index] = exp_value/row_sum;
    }

} 

torch::Tensor softmax_forward(torch::Tensor input ){
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,"input must be float32 or float16");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.size(1) == 256,"input columns must be 245");

    auto output = torch::empty_like(input);
    const int rows = static_cast<int>(input.size(0));
    const int cols = static_cast<int>(input.size(1));
    const int threads = cols;
    const int blocks = rows;
    softmax_forward_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<float>(), output.data_ptr<float>(), cols);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


torch::Tensor softmax_forward_warp(torch::Tensor input ){
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(
    input.scalar_type() == torch::kFloat32 ||
    input.scalar_type() == torch::kFloat16,
    "input must be float32 or float16"
    );
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.size(1) == 256,"input columns must be 245");

    auto output = torch::empty_like(input);
    const int rows = static_cast<int>(input.size(0));
    const int cols = static_cast<int>(input.size(1));
    const int threads = cols;
    const int blocks = rows;
    if (input.scalar_type() == torch::kFloat32) {
        softmax_forward_warp_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), cols);
    }else{
        softmax_forward_warp_half_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),reinterpret_cast<__half*>(output.data_ptr<at::Half>()),cols);
    }
    

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


torch::Tensor softmax_forward_warp_scale(torch::Tensor input,float scale ){
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(
    input.scalar_type() == torch::kFloat32 ||
    input.scalar_type() == torch::kFloat16,
    "input must be float32 or float16"
    );
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.size(1) == 256,"input columns must be 245");

    auto output = torch::empty_like(input);
    const int rows = static_cast<int>(input.size(0));
    const int cols = static_cast<int>(input.size(1));
    const int threads = cols;
    const int blocks = rows;
    if (input.scalar_type() == torch::kFloat32) {
        softmax_forward_warp_scale_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), cols,scale);
    }else{
        softmax_forward_warp_half_scale_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),reinterpret_cast<__half*>(output.data_ptr<at::Half>()),cols,scale);
    }
    

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_forward", &softmax_forward, "calcuate the softmax to a float32 CUDA 2D tensor");
  m.def("softmax_forward_warp", &softmax_forward_warp, "calcuate the softmax to a float32 CUDA 2D tensor by using warp");
  m.def("softmax_forward_warp_scale", &softmax_forward_warp_scale, "calcuate the softmax to a float32 CUDA 2D tensor by using warp scale");
}



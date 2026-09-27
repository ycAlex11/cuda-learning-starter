#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>



__global__ void residual_rmsnorm_forward_half_kernel(const __half* input,const __half* residual,const __half* weight,__half* output,int hidden_size,float eps){

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = row * hidden_size;
    float local_sum = 0.0f;
    
    for (int col = tid; col<hidden_size;col+= blockDim.x ){
        const float value = __half2float(input[base + col])+ __half2float(residual[base + col]);
        local_sum+=value*value;
    }
    __shared__ float shared[256];
    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 64; stride /= 2){
        if(tid<stride){
            shared[tid] += shared[tid+stride];
        }
        __syncthreads();
    }
    if (tid<32){
        float value = shared[tid] +shared[tid+32];
        for (int offset = 16; offset > 0; offset /= 2) {
            value +=(__shfl_down_sync(0xffffffff, value, offset));
        }
        if (tid == 0) {
        shared[0] = value;
        }
    }
    __syncthreads();
    const float sum_squares = shared[0];
    const float mean_square = sum_squares / hidden_size;
    const float rms = sqrtf(mean_square + eps);

    for(int col = tid; col<hidden_size;col+=blockDim.x){
        const float value = __half2float(input[base + col])+__half2float(residual[base + col]);
        const float weight_value = __half2float(weight[col]);
        output[base + col] = __float2half(value / rms * weight_value);
    }
}


__global__ void residual_rmsnorm_forward_kernel(
    const float* input,
    const float* residual,
    const float* weight,
    float* output,
    int hidden_size,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = row * hidden_size;
    float local_sum = 0.0f;
    
    for (int col = tid; col<hidden_size;col+= blockDim.x ){
        const float value = input[base + col] + residual[base + col];
        local_sum+=value*value;
    }
    __shared__ float shared[256];
    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 64; stride /= 2){
        if(tid<stride){
            shared[tid] += shared[tid+stride];
        }
        __syncthreads();
    }
    if (tid<32){
        float value = shared[tid] +shared[tid+32];
        for (int offset = 16; offset > 0; offset /= 2) {
            value +=(__shfl_down_sync(0xffffffff, value, offset));
        }
        if (tid == 0) {
        shared[0] = value;
        }
    }
    __syncthreads();
    const float sum_squares = shared[0];
    const float mean_square = sum_squares / hidden_size;
    const float rms = sqrtf(mean_square + eps);

    for(int col = tid; col<hidden_size;col+=blockDim.x){
        const float value = input[base + col] + residual[base + col];
        output[base + col] = value / rms * weight[col];
    }


}


__global__ void rmsnorm_forward_kernel(const float* input, const float* weight,float* output, int hidden_size,float eps ){

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = row * hidden_size;
    float local_sum = 0.0f;
    
    for (int col = tid; col<hidden_size;col+= blockDim.x ){
        const float value = input[base + col];
        local_sum+=value*value;
    }
    __shared__ float shared[256];
    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 64; stride /= 2){
        if(tid<stride){
            shared[tid] += shared[tid+stride];
        }
        __syncthreads();
    }
    if (tid<32){
        float value = shared[tid] +shared[tid+32];
        for (int offset = 16; offset > 0; offset /= 2) {
            value +=(__shfl_down_sync(0xffffffff, value, offset));
        }
        if (tid == 0) {
        shared[0] = value;
        }
    }
    __syncthreads();
    const float sum_squares = shared[0];
    const float mean_square = sum_squares / hidden_size;
    const float rms = sqrtf(mean_square + eps);

    for(int col = tid; col<hidden_size;col+=blockDim.x){
        const float value = input[base + col];
        output[base + col] = value / rms * weight[col];
    }
}


__global__ void rmsnorm_forward_half_kernel(const __half* input,const __half* weight,__half* output,int hidden_size,float eps){

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = row * hidden_size;
    float local_sum = 0.0f;
    
    for (int col = tid; col<hidden_size;col+= blockDim.x ){
        const float value = __half2float(input[base + col]);
        local_sum+=value*value;
    }
    __shared__ float shared[256];
    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 64; stride /= 2){
        if(tid<stride){
            shared[tid] += shared[tid+stride];
        }
        __syncthreads();
    }
    if (tid<32){
        float value = shared[tid] +shared[tid+32];
        for (int offset = 16; offset > 0; offset /= 2) {
            value +=(__shfl_down_sync(0xffffffff, value, offset));
        }
        if (tid == 0) {
        shared[0] = value;
        }
    }
    __syncthreads();
    const float sum_squares = shared[0];
    const float mean_square = sum_squares / hidden_size;
    const float rms = sqrtf(mean_square + eps);

    for(int col = tid; col<hidden_size;col+=blockDim.x){
        const float value = __half2float(input[base + col]);
        const float weight_value = __half2float(weight[col]);
        output[base + col] = __float2half(value / rms * weight_value);
    }
}



torch::Tensor rmsnorm_forward(torch::Tensor input,torch::Tensor weight,double eps){

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32||input.scalar_type() == torch::kFloat16,"input must be float32 or float16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32||input.scalar_type() == torch::kFloat16,"weight must be float32 or float16");
    TORCH_CHECK(input.is_contiguous()&& weight.is_contiguous(), "input and weight must be contiguous");
    const int rows = static_cast<int>(input.size(0)*input.size(1));
    const int hidden_size = static_cast<int>(input.size(2));

    auto output = torch::empty_like(input);
    if (input.scalar_type() == torch::kFloat32) {
        rmsnorm_forward_kernel<<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            hidden_size,
            static_cast<float>(eps)
        );
    } else {
       rmsnorm_forward_half_kernel<<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            hidden_size,
            static_cast<float>(eps)
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;

}


torch::Tensor residual_rmsnorm_forward(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    double eps
){

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(residual.dim() == 3, "input must be 3D");
    TORCH_CHECK(residual.scalar_type() == torch::kFloat32||input.scalar_type() == torch::kFloat16,"input must be float32 or float16");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32||input.scalar_type() == torch::kFloat16,"input must be float32 or float16");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32||input.scalar_type() == torch::kFloat16,"weight must be float32 or float16");
    TORCH_CHECK(input.is_contiguous()&& weight.is_contiguous()&&residual.is_contiguous(), "input and weight and residual must be contiguous");
    const int rows = static_cast<int>(input.size(0)*input.size(1));
    const int hidden_size = static_cast<int>(input.size(2));

    auto output = torch::empty_like(input);
    if (input.scalar_type() == torch::kFloat32) {
        residual_rmsnorm_forward_kernel<<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<float>(),
            residual.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            hidden_size,
            static_cast<float>(eps)
        );
    } else{

        residual_rmsnorm_forward_half_kernel<<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(residual.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            hidden_size,
            static_cast<float>(eps)
        );
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm_forward", &rmsnorm_forward, "calcuate the srmsnorm ");

  m.def("residual_rmsnorm_forward", &residual_rmsnorm_forward, "calcuate the srmsnorm with residual ");
}

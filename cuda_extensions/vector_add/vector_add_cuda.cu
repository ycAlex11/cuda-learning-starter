#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

__global__ void add_one_kernel(const float* input, float* output, int64_t size) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size) output[index] = input[index] + 1.0f;
}

__global__ void scale_add_kernel(const float* input,
  float* output,
  int64_t size,
  float scale,
  float bias){
    const int64_t index = static_cast<int64_t>(blockIdx.x)*blockDim.x +threadIdx.x;
    if(index<size){
      output[index] = input[index]*scale+bias;
    }
  
}

torch::Tensor add_one(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  auto output = torch::empty_like(input);
  constexpr int threads = 256;
  const int blocks = static_cast<int>((input.numel() + threads - 1) / threads);
  add_one_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<float>(), output.data_ptr<float>(), input.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor scale_add(torch::Tensor input, float scale,float bias){

  TORCH_CHECK(input.is_cuda(),"input must");
  TORCH_CHECK(
        input.scalar_type() == torch::kFloat32,
        "input must be float32"
    );
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    auto output = torch::empty_like(input);

    constexpr int threads = 256;
    const int blocks = static_cast<int>((input.numel()+threads-1)/threads);
    scale_add_kernel<<<blocks,threads,0,at::cuda::getCurrentCUDAStream()
    >>>(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      input.numel(),
      scale,
      bias
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_one", &add_one, "Add one to a float32 CUDA tensor");
  m.def(
        "scale_add",
        &scale_add,
        "Compute input * scale + bias on a float32 CUDA tensor"
    );
}

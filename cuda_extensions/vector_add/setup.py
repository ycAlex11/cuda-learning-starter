import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
setup(
    name="vector_add_cuda",
    ext_modules=[CUDAExtension(
        name="vector_add_cuda",
        sources=["vector_add_cuda.cu"],
        extra_compile_args={
            "cxx": ["/O2"],
            "nvcc": ["-O2", "--use_fast_math", "--allow-unsupported-compiler"],
        },
    )],
    cmdclass={"build_ext": BuildExtension},
)

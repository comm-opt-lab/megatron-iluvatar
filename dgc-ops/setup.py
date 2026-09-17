from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="dgc_ops_corex",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="dgc_ops_corex",
            sources=[
                "csrc/bindings.cpp",
                "csrc/dgc_ops.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-lineinfo", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

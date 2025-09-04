#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error_id) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "Not found :(" << std::endl;
    } else {
        std::cout << "Found " << deviceCount << " CUDA-enabled device(s)." << std::endl;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        std::cout << "\n--- Device " << dev << ": " << deviceProp.name << " ---" << std::endl;
        std::cout << "  Compute capability:          " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Memory:         " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Number of multiprocessors:   " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  CUDA Core count:             " << deviceProp.multiProcessorCount * 192 << " (for reference, may vary)" << std::endl;
    }

    return 0;
}
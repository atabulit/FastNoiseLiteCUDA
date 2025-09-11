# FastNoiseLiteCUDA

This is a CUDA-compatible wrapper for Auburn's popular [FastNoiseLite](https://github.com/Auburn/FastNoiseLite) library. It allows for the high-performance generation of various noise types directly on the GPU within CUDA kernels.

This port is designed to be a drop-in replacement for the original single-header file, enabling its use in both host (`__host__`) and device (`__device__`) code.

**Please note:** The primary design pattern for this library is to configure a `FastNoiseLite` instance on the host (using its setter methods) and then use it within a CUDA kernel to generate noise. While the setters are `__host__` compatible, the core noise generation functions (`GetNoise`, `DomainWarp`) are intended solely for `__device__` execution. Their implementation relies on lookup tables stored in `__constant__` memory, a read-only, cached memory space accessible only by the GPU and not the host.

## Key Modifications for CUDA Compatibility

To make the library compatible with CUDA, several key changes were made to the original source code:

*   **CUDA Function Specifiers**: All class methods and helper functions are now decorated with `__device__ __host__` (via the `NOISE_DH` macro). This allows them to be called from both CPU (host) and GPU (device) code seamlessly.

*   **Lookup Table Refactoring**: The original `Lookup` struct, which was a nested static member of the class, caused compilation issues with NVCC. The compiler struggles with the definition of static device-side members. To resolve this:
    *   The large lookup arrays (for gradients and random vectors) have been moved into a global `detail` namespace.
    *   These arrays are declared in `__constant__` memory using the `NOISE_CONSTANT` macro. Constant memory is a cached, read-only memory space on the GPU, making it highly efficient for data that is accessed uniformly by all threads in a warp.
    *   A new `FastNoise` namespace now contains the `Lookup` struct, which safely references these constant memory arrays. This restructuring resolves compilation errors while improving performance on the GPU.

*   **CUDA Math Library**: The standard `<cmath>` header has been replaced with `<cuda/cmath>` to ensure that math functions like `sqrtf` are correctly resolved to their optimized device-side implementations when compiled for the GPU.

## Usage

Include the `FastNoiseLiteCUDA.h` header in your `.cu` file. You can then instantiate and use the `FastNoiseLite` object directly inside your CUDA kernels.

### Example 1: Creating Noise Object Inside Kernel

Here is a simple example of a kernel that generates 2D OpenSimplex2 noise for a grid by creating the noise object on the device.

```cuda
#include "FastNoiseLiteCUDA.h"

__global__ void generate_noise_kernel(float* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    // Create a FastNoiseLite instance on the stack for each thread
    FastNoiseLite noise(1337); // Seed
    noise.SetNoiseType(FastNoiseLite::NoiseType_OpenSimplex2);
    noise.SetFrequency(0.05f);

    // Calculate noise value
    float noiseValue = noise.GetNoise((float)x, (float)y);

    // Write the result to the output array
    output[y * width + x] = noiseValue;
}
```

### Example 2: Configuring on Host, Passing to Kernel

A more common pattern is to configure the noise generator on the host and pass it by value to the kernel.

```cuda
#include "FastNoiseLiteCUDA.h"
#include <cuda_runtime.h>

// Kernel accepts a configured FastNoiseLite object
__global__ void generate_noise_from_host_config(float* output, int width, int height, FastNoiseLite noise)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    // Use the pre-configured noise object passed from the host
    float noiseValue = noise.GetNoise((float)x, (float)y);
    output[y * width + x] = noiseValue;
}

int main()
{
    int width = 1024;
    int height = 1024;
    size_t bufferSize = width * height * sizeof(float);

    float* d_output;
    cudaMalloc(&d_output, bufferSize);

    // 1. Configure FastNoiseLite on the host
    FastNoiseLite host_noise_generator;
    host_noise_generator.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
    host_noise_generator.SetFrequency(0.02f);
    host_noise_generator.SetFractalType(FastNoiseLite::FractalType_FBm);
    host_noise_generator.SetFractalOctaves(5);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 2. Pass the configured object by value to the kernel
    generate_noise_from_host_config<<<numBlocks, threadsPerBlock>>>(d_output, width, height, host_noise_generator);

    // ... copy data back to host and process ...

    cudaFree(d_output);
    return 0;
}
```

## Original Library

This project is a wrapper and is entirely based on the fantastic work by Jordan Peck (Auburn). All noise generation algorithms and logic belong to the original author.

For more in-depth documentation on the noise algorithms, features, and settings, please refer to the [official repository](https://github.com/Auburn/FastNoiseLite).

## License

This wrapper is distributed under the MIT License, consistent with the original FastNoiseLite library. See the [LICENSE](LICENSE) file for more detail.

#FastNoiseLiteCUDA

This is a CUDA-compatible wrapper for Auburn's popular [FastNoiseLite](https://github.com/Auburn/FastNoiseLite) library. It allows for the high-performance generation of various noise types directly on the GPU within CUDA kernels.

This port is designed to be a drop-in replacement for the original single-header file, enabling its use in both host (`__host__`) and device (`__device__`) code.

---

## *Please note: A Critical Note on Host-Side Usage*

The library's behavior changes depending on whether you are compiling a `.cu` file with NVCC or a standard `.cpp` file with a host compiler (like GCC/Clang/MSVC). This is crucial for understanding where you can safely call the `GetNoise` functions.

The reason for this is the `NOISE_CONSTANT` macro, which places large lookup tables into `__constant__` GPU memory when processed by NVCC, but compiles them as regular CPU memory otherwise.

*   **In `.cu` files (compiled with NVCC):**
    Any function that uses the lookup tables (like `GetNoise`) is prepared by NVCC for device execution. If you call such a function from host code within a `.cu` file, the CPU will try to access the `__constant__` memory on the GPU, which is an illegal operation and **will not work**.
    *   **Rule:** Inside `.cu` files, only configure `FastNoiseLite` on the host. Noise generation must happen inside a `__global__` kernel.

*   **In `.cpp` files (or other non-NVCC compiled sources):**
    When you include `FastNoiseLiteCUDA.h` in a standard `.cpp` file, `NOISE_CONSTANT` is empty. The lookup tables are just standard global arrays in CPU memory.
    *   **Rule:** Inside `.cpp` files, you can safely use the **entire** `FastNoiseLite` object on the host, including calling `GetNoise`. This allows for CPU-based noise generation for testing or other logic.

**Recommendation:**
For consistency and to avoid mistakes, the best practice is to configure your `FastNoiseLite` instance on the CPU and then pass it by value to your kernels for noise generation, as shown in *Example 2*. Use host-side `GetNoise` calls from `.cpp` files only when you have a specific need for them.

---

## Key Modifications for CUDA Compatibility

To make the library compatible with CUDA, several key changes were made to the original source code:

*   **CUDA Function Specifiers**: All class methods and helper functions are now decorated with `__device__ __host__` (via the `NOISE_DH` macro). This allows them to be called from both CPU (host) and GPU (device) code seamlessly.

*   **Lookup Table Refactoring**: The original `Lookup` struct, which was a nested static member of the class, caused compilation issues with NVCC. The compiler struggles with the definition of static device-side members. To resolve this:
    *   The large lookup arrays (for gradients and random vectors) have been moved into a global `detail` namespace.
    *   These arrays are declared in `__constant__` memory using the `NOISE_CONSTANT` macro. Constant memory is a cached, read-only memory space on the GPU, making it highly efficient for data that is accessed uniformly by all threads in a warp.
    *   A new `FastNoise` namespace now contains the `Lookup` struct, which safely references these constant memory arrays. This restructuring resolves compilation errors while improving performance on the GPU.

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
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

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

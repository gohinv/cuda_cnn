#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int W_grid = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH; // number of tiles along the width of an output feature map
    // number of tiles along height of an output feature map
    //  number of tiles per feature map
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a


    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int X_TW = TILE_WIDTH + K - 1;
    extern __shared__ float shared[];
    float* input_shared = &shared[0];
    float* mask_shared = &shared[X_TW * X_TW];
   
    int sample = blockIdx.x;
    int outmap = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int h = (blockIdx.z / W_grid) * TILE_WIDTH + ty;
    int w = (blockIdx.z % W_grid) * TILE_WIDTH + tx;

    float sum = 0;
    for (int c = 0; c < Channel; c++)
    {
        // collaboratively load mask into shared memory for current channel
        if (threadIdx.x < K && threadIdx.y < K)
        {
            mask_shared[threadIdx.y * K + threadIdx.x] = mask_4d(outmap, c, threadIdx.y, threadIdx.x);
        }
        __syncthreads();

        // load inputs into shared memory
        int bottomlim = h - ty + X_TW;
        int rightlim = w - tx + X_TW;
        for (int i = h; i < bottomlim; i+= TILE_WIDTH)
        {
            for (int j = w; j < rightlim; j+= TILE_WIDTH)
            {
                if (i < Height && j < Width)
                {
                    input_shared[(i - (h-ty)) * X_TW + (j - (w - tx))] = in_4d(sample, c, i, j);
                }
            }
        }
        __syncthreads();
        for (int p = 0; p < K; p++)
        {
            for (int q = 0; q < K; q++)
            {
                sum += input_shared[(ty+p) * X_TW + (tx + q)] * mask_shared[p*K + q];
            }
        }
        __syncthreads();
    }

    if (h < Height_out && w < Width_out)
    {
        out_4d(sample, outmap, h, w) = sum;
    }
    
    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    cudaMalloc((void**)device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void**)device_mask_ptr, Channel * Map_out * K * K * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Channel * Map_out * K * K * sizeof(float), cudaMemcpyHostToDevice);
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int W_grid = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH; // number of tiles along the width of an output feature map
    int H_grid = (Height_out + TILE_WIDTH - 1) / TILE_WIDTH; // number of tiles along height of an output feature map
    int X_TW = TILE_WIDTH + K - 1;
    size_t shared_size = ((X_TW * X_TW) + (K * K)) * sizeof(float);
    int Z = W_grid * H_grid; //  number of tiles per feature map

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(Batch, Map_out, Z);
    // size_t shared_size
    conv_forward_kernel<<<gridDim, blockDim, shared_size>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // int W_grid = Width_out / TILE_WIDTH; // number of tiles along the width of an output feature map
    // int H_grid = Height_out / TILE_WIDTH; // number of tiles along height of an output feature map
    // int Z = W_grid * H_grid; //  number of tiles per feature map

    cudaMemcpy(host_output, device_output, Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}

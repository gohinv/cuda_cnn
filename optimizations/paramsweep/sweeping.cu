#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define UNR_BLK_SIZE 256

__global__ void unroll_kernel(float* output, const float* input, const int Channel, const int Height, const int Width, const int K, int i)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    // #define in_3d(i2, i1, i0) input[(i2) * (Height * Width) + (i1) * (Width) + i0]

    // Width of unrolled input matrix
    int W_u = Height_out * Width_out;
    if (t < Channel * W_u)
    {
        // channel of the input assigned to thread
        int c = t / W_u;
        // column index of the unrolled matrix to write a strip of elements into
        // as well as linearized index of element
        int s = t % W_u;

        // Horizontal and vertical indices of the output element
        int h_out = s / Width_out;
        int w_out = s % Width_out;

        // Starting row index for the unrolled matrix section for channel c
        int base = c * K * K;
        int row_unroll = h_out * Width_out + w_out;
        // index of the unrolled matrix for the thread to write
        // the input element into for the current iteration
        for (int p = 0; p < K; p++)
        {
            for (int q = 0; q < K; q++)
            {
                int col_unroll = base + p*K + q;
                output[col_unroll * W_u + row_unroll] = input[i * Height * Width * Channel + c * Height * Width + (h_out + p) * Width + w_out + q];
            }
        }
    }
}

__global__ void multiply_kernel(float* C, const float* A, float* B, int Arows, int Acols, int Brows, int Bcols)
{
    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

    int numAColumns = Acols;
    int numARows = Arows;
    int numBColumns = Bcols;
    int numBRows = Brows;
    int numCColumns = Bcols;
    int numCRows = Arows;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    int width = ceil((float)(numAColumns)/ TILE_WIDTH);
    float sum = 0.0;

    for (int q = 0; q < width; ++q)
    {
        if ((Row < numARows) && (q * TILE_WIDTH + tx < numAColumns))
        {
            subTileA[ty][tx] = A[Row * numAColumns + q * TILE_WIDTH + tx];
        }
        else
        {
            subTileA[ty][tx] = 0;
        }
        if ((Col < numBColumns) && (q * TILE_WIDTH + ty < numBRows))
        {
            subTileB[ty][tx] = B[(q * TILE_WIDTH + ty) * numBColumns + Col];
        }
        else
        {
            subTileB[ty][tx] = 0;
        }
        __syncthreads();
        for (int n = 0; n < TILE_WIDTH; ++n)
        {
            sum += subTileA[ty][n] * subTileB[n][tx];
        }
        __syncthreads();
    }
    if ((Row < numCRows) && (Col < numCColumns))
    {
        C[Row * numCColumns + Col] = sum;
    }   
}

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
   
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    // int W_grid = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH; // number of tiles along the width of an output feature map

    // #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    // #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    // #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // // Insert your GPU convolution kernel code here
    
    // int sample = blockIdx.x;
    // int outmap = blockIdx.y;
    // int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    // int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;

    // #undef out_4d
    // #undef in_4d
    // #undef mask_4d
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

    // copy input data into global memory
    cudaMalloc((void**)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);

    // allocate device memory for final result
    cudaMalloc((void**)device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));

    // copy mask into global memory
    cudaMalloc((void**)device_mask_ptr, Channel * Map_out * K * K * sizeof(float));
    cudaMemcpy(*device_mask_ptr, host_mask, Channel * Map_out * K * K * sizeof(float), cudaMemcpyHostToDevice);
    
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    // int W_grid = (Width_out + TILE_WIDTH - 1) / TILE_WIDTH; // number of tiles along the width of an output feature map
    // int H_grid = (Height_out + TILE_WIDTH - 1) / TILE_WIDTH; // number of tiles along height of an output feature map
    // int Z = W_grid * H_grid; //  number of tiles per feature map

    // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    // dim3 gridDim(Batch, Map_out, Z);
    // conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    // cudaDeviceSynchronize();

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // allocate device memory for output of unrolling kernel
    float* input_unrolled;
    int in_unroll_h = Channel * K * K;
    int in_unroll_w = Height_out * Width_out;
    cudaMalloc((void**)&input_unrolled, in_unroll_h * in_unroll_w * sizeof(float));

    for (int b = 0; b < Batch; b++)
    {
        // make call to unroll kernel
        dim3 blockDim_u(UNR_BLK_SIZE, 1, 1);
        dim3 gridDim_u(ceil((float)(Channel * Height_out * Width_out) / UNR_BLK_SIZE), 1, 1);
        unroll_kernel<<<gridDim_u, blockDim_u>>>(input_unrolled, device_input, Channel, Height, Width, K, b);

        // make call to multiply kernel: A is masks and B is unrolled input
        int Arows = Map_out;
        int Acols = Channel * K * K;
        int Brows = Channel * K * K;
        int Bcols = Height_out * Width_out;
        int Crows = Arows;
        int Ccols = Bcols;
        dim3 DimGrid(ceil(((float)Ccols) / TILE_WIDTH), ceil(((float)Crows) / TILE_WIDTH), 1);
        dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        multiply_kernel<<<DimGrid, DimBlock>>>(device_output + b * Height_out * Width_out * Map_out, device_mask, input_unrolled, Arows, Acols, Brows, Bcols);
    }
    cudaFree(input_unrolled);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
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

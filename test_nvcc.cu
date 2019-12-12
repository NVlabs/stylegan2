// Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
//
// This work is made available under the Nvidia Source Code License-NC.
// To view a copy of this license, visit
// https://nvlabs.github.io/stylegan2/license.html

#include <cstdio>

__global__ void cudaKernel(void)
{
    printf("GPU says hello!\n");
}

int main(void)
{
    printf("CPU says hello!\n");
    cudaError_t err = cudaLaunchKernel(cudaKernel, 1, 1, NULL, 0, NULL);
    if (err != cudaSuccess)
    {
        printf("%s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

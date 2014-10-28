#include <stdio.h>

void __global__ kernel_add_one(float* a, int length) {

  int gid = threadIdx.x + blockDim.x*blockIdx.x;
  
  while(gid < length) {
    a[gid] += 1.0f;
    gid += blockDim.x*gridDim.x;
  }
  
} // kernel_add_one

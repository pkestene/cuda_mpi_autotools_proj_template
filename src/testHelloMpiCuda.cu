/**
 * \file testHelloMpiCuda.cu
 * \brief A simple program to test MPI+Cuda
 * 
 * For a detailled discussion about coupling CUDA and MPI, see the following references:
 * - https://github.com/parallel-forall/code-samples/tree/master/posts/cuda-aware-mpi-example
 * - http://www.open-mpi.org/faq/?category=running#mpi-cuda-support
 *
 * Here, were use the simple approach : init CUDA device after MPI environment is
 * initialized.
 *
 * \date 8 Oct 2010
 * \author Pierre Kestener
 */

// MPI-related includes
#include <mpi.h>

// CUDA-C includes
#include <cuda_runtime_api.h>

// CUDA kernel
#include "kernel.cu"

#include <cstdio>
#include <cassert>

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
  
  // initialize MPI session
  MPI_Init(&argc,&argv);
  
  int myMpiRank, nMpiProc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myMpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nMpiProc);

  // initialize cuda
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
    printf("cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n");
    printf("\nFAILED\n");
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0)
    printf("There is no device supporting CUDA\n");

  // grab information about current GPU device
  cudaDeviceProp deviceProp;
  int deviceId;
  int driverVersion = 0, runtimeVersion = 0;
  cudaSetDevice(myMpiRank%deviceCount);
  cudaGetDevice(&deviceId);
  cudaGetDeviceProperties(&deviceProp, deviceId);

  // grab information about CPU node / MPI process:
  int nameLen;
  char procName[MPI_MAX_PROCESSOR_NAME+1];
  int mpierr = MPI_Get_processor_name(procName,&nameLen);

  // dump information
  if (myMpiRank >= 0) {
    if (deviceProp.major == 9999 && deviceProp.minor == 9999)
      printf("There is no device supporting CUDA.\n");
    else
      printf("There are %d devices supporting CUDA associated with MPI process of rank %d\n", deviceCount,myMpiRank);

    printf("Using Device %d: \"%s\"\n", deviceId, deviceProp.name);

#if CUDART_VERSION >= 2020

    // Console log
    cudaDriverGetVersion(&driverVersion);
    printf("  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
#endif

    printf("  CUDA Capability Major revision number:         %d\n", deviceProp.major);
    printf("  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);
    
    printf("  Total amount of global memory:                 %lu bytes (%lu MBytes)\n", deviceProp.totalGlobalMem, deviceProp.totalGlobalMem/1024/1024);

  }

  printf("MPI process number %d out of %d on machine %s\nWorking GPU device Id is %d\n",myMpiRank,nMpiProc,procName,deviceId);

  // execute a CUDA kernel
  {
    int size = 100000;
    size_t sizeBytes = size*sizeof(float);

    float* data_cpu = (float*) malloc(sizeBytes);
    float* data_gpu;
    cudaMalloc((void**) &data_gpu, sizeBytes);

    // init data on CPU
    for (int i=0; i<size; i++)
      data_cpu[i] = 2.0*i+myMpiRank;

    // upload data to GPU
    cudaMemcpy(data_gpu, data_cpu, sizeBytes, cudaMemcpyHostToDevice);

    // compute: ten blocks of 256 threads each
    kernel_add_one<<<10,256>>>(data_gpu, size);

    // download result and check
    cudaMemcpy(data_cpu, data_gpu, sizeBytes, cudaMemcpyDeviceToHost);
    for (int i=0; i<size; i++)
      assert (data_cpu[i] == 2.0*i+1.0+myMpiRank);

    printf("Test OK on MPI rank %d\n",myMpiRank);

    // free memory
    free(data_cpu);
    cudaFree(data_gpu);

  } // end execute a CUDA kernel


  // MPI finalize
  MPI_Finalize();

  return EXIT_SUCCESS;

}

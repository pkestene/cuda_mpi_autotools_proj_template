cuda_mpi_autotools_proj_template
================================

A template project for CUDA+MPI with autotools build system

## Build for the first time

### Run autogen.sh script. 

> ./autogen.sh

It will create all autotools related files.

### Run configure

> ./configure --with-cuda=$CUDA_ROOT --enable-mpi

where CUDA_ROOT is an environment variable set to the location of the CUDA toolkit (e.g. /usr/local/cuda-6.5)

### Run make

This will build an executable src/testHelloMpiCuda.

### Start application

You can run it on a multi-GPU cluster with

> mpirun -np N_PROC_MPI ./testHelloMpiCuda


## Reference

For a detailled discussion about coupling CUDA and MPI (initialize CUDA context before/after MPI_Init), see the following references:


- https://github.com/parallel-forall/code-samples/tree/master/posts/cuda-aware-mpi-example
- http://www.open-mpi.org/faq/?category=running#mpi-cuda-support

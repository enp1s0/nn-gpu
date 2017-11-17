#pragma once

#define CUDA_HANDLE_ERROR( err ) (cudaHandleError( err, __FILE__, __LINE__ ))

void cudaHandleError( cudaError_t err,const char *file,int line );

__host__ __device__ inline int threads_ceildiv(int size,int blocks);


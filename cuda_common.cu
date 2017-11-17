#include <stdio.h>
#include <stdlib.h>

void cudaHandleError( cudaError_t err,const char *file,int line ) {
	if (err != cudaSuccess) {
		printf( "CUDA Error\n%s in %s at line %d\n", cudaGetErrorString( err ),file, line );
		exit( EXIT_FAILURE );
	}
}
__host__ __device__ inline int threads_ceildiv(int size,int blocks){
	return (blocks + size - 1)/blocks;
}

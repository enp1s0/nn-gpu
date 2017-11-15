#include "matrix_array.h"
#include "cuda_common.h"
#include <curand.h>
#include <curand_kernel.h>
#include <random>

using namespace mtk;

const int BLOCKS = 1 << 7;


__global__ void deviceSetConstant(float *device_ptr,float f,int max_t){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(max_t <= tid)
		return;
	device_ptr[tid] = f;
}
__global__ void deviceSetRandom(float *device_ptr,float min,float max,int seed,int max_t){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(max_t <= tid)
		return;
	
	curandState s;
	curand_init(seed,tid,0,&s);
	device_ptr[tid] = curand_uniform(&s) * (max - min) + min;
}
__global__ void deviceCopy(float* device_ptr_dst,float* device_ptr_src,int max_t){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(max_t <= tid)
		return;
	device_ptr_dst[tid] = device_ptr_src[tid];
}

MatrixXf::MatrixXf(int rows,int cols):
	rows(rows),cols(cols),device_ptr(nullptr),host_ptr(nullptr)
{}

MatrixXf::MatrixXf():MatrixXf(0,0)
{}

MatrixXf::~MatrixXf(){
	CUDA_HANDLE_ERROR( cudaFree( device_ptr ) );
	CUDA_HANDLE_ERROR( cudaFreeHost( host_ptr ) );
	device_ptr = nullptr;
	host_ptr = nullptr;
}

MatrixXf* MatrixXf::setSize(int rows,int cols){
	this->rows = rows;
	this->cols = cols;
	return this;
}

MatrixXf* MatrixXf::allocateDevice(){
	CUDA_HANDLE_ERROR( cudaMalloc( (void**)&device_ptr, sizeof(float) * rows * cols ));
	return this;
}

MatrixXf* MatrixXf::allocateHost(){
	CUDA_HANDLE_ERROR( cudaMallocHost( (void**)&host_ptr, sizeof(float) * rows * cols ));
	return this;
}


void MatrixXf::copyToDevice(){
	CUDA_HANDLE_ERROR( cudaMemcpy( device_ptr, host_ptr, sizeof(float) * rows * cols, cudaMemcpyHostToDevice ) );
}

void MatrixXf::copyToHost(){
	CUDA_HANDLE_ERROR( cudaMemcpy( host_ptr, device_ptr, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost ) );
}

void MatrixXf::copyTo(float* dst_ptr)const{
	deviceCopy<<<BLOCKS,(rows*cols+BLOCKS-1)/BLOCKS>>>(dst_ptr,device_ptr,rows*cols);
}
void MatrixXf::copyTo(mtk::MatrixXf& matrix)const{
	copyTo(matrix.getDevicePointer());
}



int MatrixXf::getCols()const{return cols;}
int MatrixXf::getRows()const{return rows;}

float* MatrixXf::getDevicePointer()const{return device_ptr;}
float* MatrixXf::getHostPointer()const{return host_ptr;}

void MatrixXf::operator=(MatrixXf m){
	this->cols = m.getCols();
	this->rows = m.getRows();
}

void MatrixXf::initDeviceConstant(float f){
	deviceSetConstant<<<BLOCKS,(rows*cols+BLOCKS-1)/BLOCKS>>>(device_ptr,f,rows*cols);
}

void MatrixXf::initDeviceRandom(float min,float max){
	std::random_device random;
	deviceSetRandom<<<BLOCKS,(rows*cols+BLOCKS-1)/BLOCKS>>>(device_ptr,min,max,random(),rows*cols);
}

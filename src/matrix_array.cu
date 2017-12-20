#include "matrix_array.h"
#include "cuda_common.h"
#include <curand.h>
#include <curand_kernel.h>
#include <random>
#include <iostream>
#include <iomanip>

using namespace mtk;

const int BLOCKS = 1 << 7;


__global__ void deviceSetConstant(float *device_ptr,float f,int w,int max_t){
	int tid = (threadIdx.x + blockIdx.x * blockDim.x)*w;
	for(int i = 0;i < w;i++){
		if(max_t <= tid)
			return;
		device_ptr[tid] = f;
		tid++;
	}
}
__global__ void deviceSetRandom(float *device_ptr,float min,float max,int seed,int w,int max_t){
	int tid = (threadIdx.x + blockIdx.x * blockDim.x)*w;
	curandState s;
	curand_init(seed,tid,0,&s);
	for(int i = 0;i < w;i++){
		if(max_t <= tid)
			return;
		device_ptr[tid] = curand_uniform(&s) * (max - min) + min;
		tid++;
	}
}
__global__ void deviceCopy(float* device_ptr_dst,float* device_ptr_src,int max_t){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(max_t <= tid)
		return;
	device_ptr_dst[tid] = device_ptr_src[tid];
}

MatrixXf::MatrixXf(int rows,int cols):
	rows(rows),cols(cols),device_ptr(nullptr),host_ptr(nullptr),depth(0)
{}

MatrixXf::MatrixXf():MatrixXf(0,0)
{}

MatrixXf::~MatrixXf(){
	this->releaseDevice();
	this->releaseHost();
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


MatrixXf* MatrixXf::copyToDevice(){
	CUDA_HANDLE_ERROR( cudaMemcpy( device_ptr, host_ptr, sizeof(float) * rows * cols, cudaMemcpyHostToDevice ) );
	return this;
}

MatrixXf* MatrixXf::copyToHost(){
	CUDA_HANDLE_ERROR( cudaMemcpy( host_ptr, device_ptr, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost ) );
	return this;
}

MatrixXf* MatrixXf::copyTo(float* dst_ptr){
	deviceCopy<<<BLOCKS,(rows*cols+BLOCKS-1)/BLOCKS>>>(dst_ptr,device_ptr,rows*cols);
	return this;
}
MatrixXf* MatrixXf::copyTo(mtk::MatrixXf& matrix){
	copyTo(matrix.getDevicePointer());
	return this;
}



int MatrixXf::getCols()const{return cols;}
int MatrixXf::getRows()const{return rows;}
int MatrixXf::getSize()const{return rows * cols;}

float* MatrixXf::getDevicePointer()const{return device_ptr;}
float* MatrixXf::getHostPointer()const{return host_ptr;}
void MatrixXf::setDevicePointer(float* dp){device_ptr = dp;}
void MatrixXf::setHostPointer(float* hp){host_ptr = hp;}

void MatrixXf::operator=(MatrixXf m){
	this->cols = m.getCols();
	this->rows = m.getRows();
}

MatrixXf* MatrixXf::initDeviceConstant(float f){
	deviceSetConstant<<<BLOCKS,std::min(512,threads_ceildiv(rows*cols,BLOCKS))>>>(device_ptr,f,(threads_ceildiv(rows*cols,BLOCKS)+511)/512,rows*cols);
	CUDA_HANDLE_ERROR(cudaDeviceSynchronize());
	return this;
}

MatrixXf* MatrixXf::initDeviceRandom(float min,float max){
	std::random_device random;
	deviceSetRandom<<<BLOCKS,std::min(512,threads_ceildiv(rows*cols,BLOCKS))>>>(device_ptr,min,max,random(),(threads_ceildiv(rows*cols,BLOCKS)+511)/512,rows*cols);
	CUDA_HANDLE_ERROR(cudaDeviceSynchronize());
	return this;
}

MatrixXf* MatrixXf::print(std::string label){
	if(label.compare("") != 0)
		std::cout<<label<<" = "<<std::endl;
	for(int i = 0;i < rows;i++){
		for(int j = 0;j < cols;j++){
			if(host_ptr[j * rows + i] >= 0.0f)
				printf(" %.3f ",host_ptr[j * rows + i]);
			else
				printf("%.3f ",host_ptr[j * rows + i]);
			//std::cout<<std::setw(5)<<host_ptr[j * rows + i]<<" ";
		}
		std::cout<<std::endl;
	}
	return this;
}

MatrixXf* MatrixXf::releaseDevice(){
	if(depth==0){
		CUDA_HANDLE_ERROR( cudaFree( device_ptr ) );
		device_ptr = nullptr;
	}
	return this;
}
MatrixXf* MatrixXf::releaseHost(){
	if(depth==0){
		CUDA_HANDLE_ERROR( cudaFreeHost( host_ptr ) );
		host_ptr = nullptr;
	}
	return this;
}

MatrixXf* MatrixXf::splitDevice(mtk::MatrixXf& s0_mat,mtk::MatrixXf& s1_mat){
	if(s0_mat.getSize()+s1_mat.getSize() < this->getSize()){
		return this;
	}
	if(device_ptr == nullptr){
		return this;
	}
	//s0_mat.device_ptr = device_ptr;
	s0_mat.depth = depth + 1;
	s0_mat.setDevicePointer(device_ptr);
	//s1_mat.device_ptr = device_ptr + s0_mat.getSize();
	s1_mat.setDevicePointer(device_ptr + s0_mat.getSize());
	s1_mat.depth = depth + 1;
	return this;
}
MatrixXf* MatrixXf::splitHost(mtk::MatrixXf& s0_mat,mtk::MatrixXf& s1_mat){
	if(s0_mat.getSize()+s1_mat.getSize() < this->getSize()){
		return this;
	}
	if(host_ptr == nullptr){
		return this;
	}
	//s0_mat.host_ptr = host_ptr;
	s0_mat.setHostPointer(host_ptr);
	s0_mat.depth = depth + 1;
	s1_mat.setHostPointer(host_ptr + s0_mat.getSize());
	//s1_mat.host_ptr = host_ptr	+ s0_mat.getSize();
	s1_mat.depth = depth + 1;
	return this;
}

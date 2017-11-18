#include "hiddenlayer.h"
#include "cuda_common.h"
#include "activation.h"
#include "matrix_function.h"
#include <iostream>

using namespace mtk;

const int BLOCKS = 1 << 7;

template<class T>
__global__ void deviceMap(float *device_ptr_dst,float* device_ptr_src,int max_t){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(max_t <= tid)
		return;
	device_ptr_dst[tid] = T()(device_ptr_src[tid]);
}
__global__ void devicePointwiseProduct(float *device_ptr_dst,float* device_ptr_src0,float* device_ptr_src1,int max_t){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(max_t <= tid)
		return;
	device_ptr_dst[tid] = device_ptr_src0[tid] * device_ptr_src1[tid];
}




HiddenLayer::HiddenLayer(int input_size,int output_size,int batch_size,std::string layer_name,cublasHandle_t cublas,float learning_rate,float adagrad_epsilon,float annuation_rate):
	BaseLayer(input_size,output_size,batch_size,layer_name,cublas,learning_rate,adagrad_epsilon,annuation_rate)
{
}

HiddenLayer::~HiddenLayer(){
}
//HiddenLayer::~HiddenLayer(){}

void HiddenLayer::learningBackPropagation(mtk::MatrixXf &next_error, const mtk::MatrixXf &d2, const mtk::MatrixXf *w2){
	int u1_size = u1.getRows() * u1.getCols();
	const float one = 1.0f,zero = 0.0f;
	deviceMap<dActReLU><<<BLOCKS,threads_ceildiv(u1.getSize(),BLOCKS)>>>(u1.getDevicePointer(),u1.getDevicePointer(),u1.getSize());
	CUBLAS_HANDLE_ERROR(cublasSgemm(cublas,CUBLAS_OP_T,CUBLAS_OP_N,
			output_size,batch_size,w2->getRows(),
			&one,
			w2->getDevicePointer(),w2->getRows(),
			d2.getDevicePointer(),d2.getRows(),
			&zero,
			u.getDevicePointer(),u.getRows()));
	mtk::MatrixFunction::elementwiseProduct(cublas,next_error,u,u1);
	float alpha = 1.0f/batch_size;
	CUBLAS_HANDLE_ERROR(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_T,
			output_size,input_size,batch_size,
			&alpha,
			next_error.getDevicePointer(),next_error.getRows(),
			z0.getDevicePointer(),z0.getRows(),
			&zero,
			rdw1.getDevicePointer(),rdw1.getRows()));
	CUBLAS_HANDLE_ERROR(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
			output_size,1,batch_size,
			&alpha,
			next_error.getDevicePointer(),next_error.getRows(),
			all1_b.getDevicePointer(),z0.getRows(),
			&zero,
			rdb1.getDevicePointer(),rdb1.getRows()));
}

void HiddenLayer::activation(mtk::MatrixXf &output, const mtk::MatrixXf &input) const {
	deviceMap<dActReLU><<<BLOCKS,threads_ceildiv(input.getSize(),BLOCKS)>>>(output.getDevicePointer(),input.getDevicePointer(),input.getSize());
}

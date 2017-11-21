#include "hiddennetwork.h"
#include "cuda_common.h"
#include "activation.h"
#include "matrix_function.h"
#include <iostream>

using namespace mtk;


HiddenNetwork::HiddenNetwork(int input_size,int output_size,int batch_size,std::string network_name,cublasHandle_t cublas,float learning_rate,float adagrad_epsilon,float annuation_rate):
	BaseNetwork(input_size,output_size,batch_size,network_name,cublas,learning_rate,adagrad_epsilon,annuation_rate)
{
}

HiddenNetwork::~HiddenNetwork(){}

void HiddenNetwork::learningBackPropagation(mtk::MatrixXf &next_error, const mtk::MatrixXf &d2, const mtk::MatrixXf *w2){
	const float one = 1.0f,zero = 0.0f;
	mtk::MatrixFunction::map<dActReLU>(u1,u1);
	CUDA_HANDLE_ERROR(cudaDeviceSynchronize());
	CUBLAS_HANDLE_ERROR(cublasSgemm(cublas,CUBLAS_OP_T,CUBLAS_OP_N,
			u.getRows(),u.getCols(),d2.getRows(),
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
			all1_b.getDevicePointer(),batch_size,
			&zero,
			rdb1.getDevicePointer(),rdb1.getRows()));
}
void HiddenNetwork::learningActivation(mtk::MatrixXf &output, const mtk::MatrixXf &input) {
	mtk::MatrixFunction::map<ActReLU>(output,input);
}
void HiddenNetwork::testActivation(mtk::MatrixXf &output, const mtk::MatrixXf &input) {
	mtk::MatrixFunction::map<ActReLU>(output,input);
}

void HiddenNetwork::learningBackPropagation(mtk::MatrixXf &next_error, const mtk::MatrixXf &d2){
	BaseNetwork::learningBackPropagation(next_error,d2);
}

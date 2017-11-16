#include <iostream>
#include "cuda_common.h"
#include "hiddenlayer.h"
#include "matrix_array.h"
#include "softmaxlayer.h"


int main(){
	cublasHandle_t cublas;
	CUBLAS_HANDLE_ERROR(cublasCreate(&cublas));
	mtk::SoftmaxLayer s0(100,10,10,"output layer",cublas);
	mtk::MatrixXf input,output;
	input.setSize(100,10)->allocateDevice()->initDeviceRandom(-1.0f,1.0f);
	output.setSize(10,10)->allocateDevice()->allocateHost()->initDeviceConstant(0.0f);
	s0.learningForwardPropagation(output,input);
	output.copyToHost();
	CUBLAS_HANDLE_ERROR(cublasDestroy( cublas));
}

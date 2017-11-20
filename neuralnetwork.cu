#include "neuralnetwork.h"
#include "cublas_common.h"
#include "matrix_function.h"

using namespace mtk;

NeuralNetwork::NeuralNetwork(int batch_size,cublasHandle_t cublas)
	: batch_size(batch_size),cublas(cublas)
{}

NeuralNetwork* NeuralNetwork::add(mtk::BaseNetwork *network){
	networks.push_back(network);
	return this;
}

NeuralNetwork* NeuralNetwork::calcError(mtk::MatrixXf &error,const mtk::MatrixXf &output,const mtk::MatrixXf& teacher){
	float minus_one = -1.0f;
	mtk::MatrixFunction::copy(cublas,error,output);
	CUBLAS_HANDLE_ERROR(cublasSaxpy(cublas,output.getSize(), &minus_one,
				teacher.getDevicePointer(),1,
				error.getDevicePointer(),1));
	return this;
}

NeuralNetwork* NeuralNetwork::construct(){
	return this;
}

NeuralNetwork* NeuralNetwork::learningForwardPropagation(mtk::MatrixXf& output,const mtk::MatrixXf& input){
	return this;
}

NeuralNetwork* NeuralNetwork::learningBackPropagation(const mtk::MatrixXf &error){
	return this;
}

void NeuralNetwork::release(){
	for(auto network : networks){
		delete network;
	}
}

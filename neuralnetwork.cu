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

NeuralNetwork::~NeuralNetwork(){
	this->release();
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
	for(int i = 0;i < networks.size()-1;i++){
		mtk::MatrixXf *layer = new mtk::MatrixXf;
		layer->setSize(networks[i]->getOutputSize(),batch_size)->allocateDevice()->initDeviceConstant(0.0f);
		layers.push_back(layer);
		mtk::MatrixXf *error = new mtk::MatrixXf;
		error->setSize(networks[i+1]->getOutputSize(),batch_size)->allocateDevice()->initDeviceConstant(0.0f);
		errors.push_back(error);
	}
	return this;
}

NeuralNetwork* NeuralNetwork::learningForwardPropagation(mtk::MatrixXf& output,const mtk::MatrixXf& input){
	if(networks.size()==1){
		networks[0]->learningForwardPropagation(output,input);
	}else{
		networks[0]->learningForwardPropagation(*(layers[0]),input);
		for(int i = 1;i < networks.size()-2;i++){
			networks[i]->learningForwardPropagation((*layers[i]),(*layers[i-1]));
		}
		networks[networks.size()-1]->learningForwardPropagation(output,(*layers[networks.size()-2]));
	}
	return this;
}

NeuralNetwork* NeuralNetwork::learningBackPropagation(const mtk::MatrixXf &error){
	networks[networks.size()-1]->learningBackPropagation((*errors[errors.size()-1]),error,nullptr);
	for(int i = networks.size()-2;i >= 0;i--){
		networks[i]->learningBackPropagation(*errors[i],*errors[i+1],networks[i+1]->getWeightPointer());
	}
	for(auto network : networks)
		network->learningReflect();
	return this;
}

void NeuralNetwork::release(){
	for(auto network : networks){
		delete network;
	}
	for(auto layer : layers){
		delete layer;
	}
	for(auto error : errors){
		delete error;
	}
}

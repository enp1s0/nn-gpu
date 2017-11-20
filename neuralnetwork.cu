#include "neuralnetwork.h"
#include "cublas_common.h"

using namespace mtk;

NeuralNetwork::NeuralNetwork(int batch_size)
	: batch_size(batch_size)
{}

//NeuralNetwork* NeuralNetwork::add(int input_size,int output_size,std::string network_name,float learning_rate,float adagrad_epsilon,float attenuation_rate){
NeuralNetwork* NeuralNetwork::add(mtk::BaseNetwork *network){
	//Network *network = new Network(input_size,output_size,batch_size,network_name,learning_rate,adagrad_epsilon,attenuation_rate);
	//networks.push_back(network);
	return this;
}

NeuralNetwork* NeuralNetwork::calcError(mtk::MatrixXf &error,const mtk::MatrixXf &output,const mtk::MatrixXf& teacher){
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

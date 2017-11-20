#pragma once
#include "basenetwork.h"
#include "hyperparameter.h"
#include <vector>

namespace mtk{
	class NeuralNetwork{
		int batch_size;
		std::vector<mtk::MatrixXf*> layers;
		std::vector<mtk::MatrixXf*> errors;
		std::vector<mtk::BaseNetwork*> networks;
	public:
		NeuralNetwork(int batch_size);
		NeuralNetwork* add(mtk::BaseNetwork* network);
		//NeuralNetwork* add(int input_size,int output_size,std::string network_name,float learning_rate,float adagrad_epsilon,float attenuation_rate);
		NeuralNetwork* calcError(mtk::MatrixXf& error,const mtk::MatrixXf& output,const mtk::MatrixXf& teacher);
		NeuralNetwork* learningForwardPropagation(mtk::MatrixXf& output,const mtk::MatrixXf& input);
		NeuralNetwork* learningBackPropagation(const mtk::MatrixXf& error);
		void release();
	};
}

#pragma once
#include "basenetwork.h"
#include "hyperparameter.h"
#include <vector>

namespace mtk{
	class NeuralNetwork{
		int batch_size;
		cublasHandle_t cublas;
		std::vector<mtk::MatrixXf*> layers;
		std::vector<mtk::MatrixXf*> errors;
		std::vector<mtk::BaseNetwork*> networks;
	public:
		NeuralNetwork(int batch_size,cublasHandle_t cubals);
		NeuralNetwork* add(mtk::BaseNetwork* network);
		NeuralNetwork* construct();
		NeuralNetwork* calcError(mtk::MatrixXf& error,const mtk::MatrixXf& output,const mtk::MatrixXf& teacher);
		NeuralNetwork* learningForwardPropagation(mtk::MatrixXf& output,const mtk::MatrixXf& input);
		NeuralNetwork* learningBackPropagation(const mtk::MatrixXf& error);
		void release();
	};
}

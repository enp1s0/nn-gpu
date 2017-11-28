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
		std::vector<mtk::BaseUnit*> networks;
		std::vector<mtk::MatrixXf*> test_layers;
	public:
		NeuralNetwork(int batch_size,cublasHandle_t cubals);
		~NeuralNetwork();
		NeuralNetwork* add(mtk::BaseUnit* network);
		NeuralNetwork* construct();
		NeuralNetwork* calcError(mtk::MatrixXf& error,const mtk::MatrixXf& output,const mtk::MatrixXf& teacher);
		NeuralNetwork* learningForwardPropagation(mtk::MatrixXf& output,const mtk::MatrixXf& input);
		NeuralNetwork* learningBackPropagation(const mtk::MatrixXf& error);
		void release();

		// test
		NeuralNetwork* testInit(int test_batch_size);
		NeuralNetwork* testForwardPropagation(mtk::MatrixXf& output,const mtk::MatrixXf &input);
		void testRelease();
	};
}

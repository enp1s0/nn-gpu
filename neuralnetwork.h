#pragma once
#include "baselayer.h"
#include <vector>

namespace mtk{
	class NeuralNetwork{
		int batch_size;
	public:
		NeuralNetwork(int batch_size);
		template<mtk::BaseLayer layer>
		NeuralNetwork* add(int input_size,int output_size,std::string layer_name,float learning_rate,float adagrad_epsilon,float attenuation_rate);

	};
}

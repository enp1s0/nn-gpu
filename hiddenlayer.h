#pragma once
#include "baselayer.h"
#include "hyperparameter.h"

namespace mtk{

	class HiddenLayer : public mtk::BaseLayer{
		void activation(mtk::MatrixXf& output,const mtk::MatrixXf& input) const;
	public:
		HiddenLayer(int input_size,int output_size,int batch_size,std::string layer_name,cublasHandle_t cublas,float learning_rate = default_learning_rate,float adagrad_epsilon = default_adagrad_epsilon,float attenuation_rate = default_attenuation_rate);
		~HiddenLayer();
		void learningBackPropagation(mtk::MatrixXf& next_error,const mtk::MatrixXf &d2,const mtk::MatrixXf* w2);
	};

}
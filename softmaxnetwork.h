#pragma once

#include "basenetwork.h"
#include "hyperparameter.h"
namespace mtk{
	class SoftmaxNetwork : public BaseNetwork{
		mtk::MatrixXf input_row_0;
		mtk::MatrixXf inverse;
		mtk::MatrixXf output0;
		void activation(mtk::MatrixXf& output,const mtk::MatrixXf& input) ;
	public:
		SoftmaxNetwork(int input_size,int output_size,int batch_size,std::string network_name,cublasHandle_t cublas,float learning_rate = default_learning_rate,float adagrad_epsilon = default_adagrad_epsilon,float attenuation_rate = default_attenuation_rate);
		~SoftmaxNetwork();
		void learningBackPropagation(mtk::MatrixXf& next_error,const mtk::MatrixXf& d2,const mtk::MatrixXf* w2 = nullptr);
	};
}

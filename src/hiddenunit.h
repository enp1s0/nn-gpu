#pragma once
#include "basenetwork.h"
#include "hyperparameter.h"

namespace mtk{

	class HiddenUnit : public mtk::BaseUnit{
		void learningActivation(mtk::MatrixXf& output,const mtk::MatrixXf& input);
		void testActivation(mtk::MatrixXf& output,const mtk::MatrixXf& input);
	public:
		HiddenUnit(int input_size,int output_size,int batch_size,std::string unit_name,cublasHandle_t cublas,float learning_rate = default_learning_rate,float adagrad_epsilon = default_adagrad_epsilon,float attenuation_rate = default_attenuation_rate);
		~HiddenUnit();
		void learningBackPropagation(mtk::MatrixXf& next_error,const mtk::MatrixXf &d2,const mtk::MatrixXf* w2);
		void learningBackPropagation(mtk::MatrixXf& next_error,const mtk::MatrixXf &d2);
	};

}

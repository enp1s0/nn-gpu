#pragma once

#include "baseunit.h"
#include "hyperparameter.h"
namespace mtk{
	class SoftmaxUnit : public BaseUnit{
		mtk::MatrixXf input_row_0;
		mtk::MatrixXf inverse;
		mtk::MatrixXf output0;
		mtk::MatrixXf test_input_row_0;
		mtk::MatrixXf test_inverse;
		mtk::MatrixXf test_output0;
		void testActivation(mtk::MatrixXf& output,const mtk::MatrixXf& input) ;
		void learningActivation(mtk::MatrixXf& output,const mtk::MatrixXf& input) ;
	public:
		SoftmaxUnit(int input_size,int output_size,int batch_size,std::string unit_name,cublasHandle_t cublas,float learning_rate = default_learning_rate,float adagrad_epsilon = default_adagrad_epsilon,float attenuation_rate = default_attenuation_rate);
		~SoftmaxUnit();
		void learningBackPropagation(mtk::MatrixXf& next_error,const mtk::MatrixXf& d2,const mtk::MatrixXf* w2);
		void learningBackPropagation(mtk::MatrixXf& next_error,const mtk::MatrixXf& d2);

		void testInit(int test_batch_size);
		void testRelease();
	};
}

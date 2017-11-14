#pragma once
#include <string>
#include "matrix_array.h"
#include "cublas_common.h"
namespace mtk{
	class BaseLayer{
		std::string layer_name;
		int output_size,input_size;
		int batch_size;
		cublasHandle_t *cublas;

		mtk::MatrixXf w1;
		mtk::MatrixXf dw1;
		mtk::MatrixXf rdw1;
		mtk::MatrixXf b1;
		mtk::MatrixXf db1;
		mtk::MatrixXf rdb1;
		mtk::MatrixXf u1;
		mtk::MatrixXf z0;
		mtk::MatrixXf d1;
		mtk::MatrixXf adagrad_w1;
		mtk::MatrixXf adagrad_b1;

		mtk::MatrixXf all1_b;

	public:
		BaseLayer(int input_size,int output_size,int batch_size,std::string layer_name,cublasHandle_t* cublas);
		~BaseLayer();
		void learningForwardPropagate(mtk::MatrixXf &output,const mtk::MatrixXf &input);
		void learningReflect();
		virtual void learningBackPropagate(mtk::MatrixXf& next_error,const mtk::MatrixXf &d2,const mtk::MatrixXf* w2) = 0;

		void testForwardPropagate(mtk::MatrixXf &output,const mtk::MatrixXf &input) const;

		void activation(mtk::MatrixXf& output,const mtk::MatrixXf& input) const;

		mtk::MatrixXf* getWeightPointer();
		mtk::MatrixXf* getBiasPointer();

	};
}

#pragma once

#include "baselayer.h"
namespace mtk{
	class SoftmaxLayer : public BaseLayer{
		mtk::MatrixXf input_row_0;
		mtk::MatrixXf all1_u;
		mtk::MatrixXf inverse;
		mtk::MatrixXf output0;
		void activation(mtk::MatrixXf& output,const mtk::MatrixXf& input) const;
	public:
		SoftmaxLayer(int input_size,int output_size,int batch_size,std::string layer_name,cublasHandle_t *cublas);
		~SoftmaxLayer();
		void learningBackPropagation(mtk::MatrixXf& next_error,const mtk::MatrixXf& d2,const mtk::MatrixXf* w2);
	};
}

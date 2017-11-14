#pragma once
#include "baselayer.h"

namespace mtk{

	class HiddenLayer : public mtk::BaseLayer{
		void activation(mtk::MatrixXf& output,const mtk::MatrixXf& input) const;
	public:
		HiddenLayer(int input_size,int output_size,int batch_size,std::string layer_name,cublasHandle_t* cublas);
		~HiddenLayer();
		void learningBackPropagation(mtk::MatrixXf& next_error,const mtk::MatrixXf &d2,const mtk::MatrixXf* w2);
	};

}

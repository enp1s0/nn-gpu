#include "hiddenlayer.h"

using namespace mtk;

HiddenLayer::HiddenLayer(int input_size,int output_size,int batch_size,std::string layer_name,cublasHandle_t *cublas):
	BaseLayer(input_size,output_size,batch_size,layer_name,cublas)
{}

HiddenLayer::~HiddenLayer(){}

void HiddenLayer::learningBackPropagation(mtk::MatrixXf &next_error, const mtk::MatrixXf &d2, const mtk::MatrixXf *w2){

}

void HiddenLayer::activation(mtk::MatrixXf &output, const mtk::MatrixXf &input) const {

}

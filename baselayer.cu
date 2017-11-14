#include "baselayer.h"

using namespace mtk;

BaseLayer::BaseLayer(int input_size,int output_size,int batch_size,std::string layer_name):
	input_size(input_size),output_size(output_size),batch_size(batch_size),layer_name(layer_name)
{
	w1.setSize(output_size,input_size)->allocateDevice()->initDeviceConstant(0.0f);
	dw1.setSize(output_size,input_size)->allocateDevice()->initDeviceConstant(0.0f);
	rdw1.setSize(output_size,input_size)->allocateDevice()->initDeviceConstant(0.0f);
	b1.setSize(output_size,1)->allocateDevice()->initDeviceRandom(-1.0f,1.0f);
	db1.setSize(output_size,1)->allocateDevice()->initDeviceConstant(0.0f);
	u1.setSize(output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	z0.setSize(input_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	adagrad_w1.setSize(output_size,input_size)->allocateDevice()->initDeviceConstant(0.0f);
	adagrad_b1.setSize(output_size,1)->allocateDevice()->initDeviceConstant(0.0f);
}

BaseLayer::~BaseLayer(){}

void BaseLayer::learningForwardPropagate(mtk::MatrixXf &output,const mtk::MatrixXf& input){
	
}

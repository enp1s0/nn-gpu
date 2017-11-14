#include "baselayer.h"

using namespace mtk;

BaseLayer::BaseLayer(int input_size,int output_size,int batch_size,std::string layer_name,cublasHandle_t* cublas):
	input_size(input_size),output_size(output_size),batch_size(batch_size),layer_name(layer_name),cublas(cublas)
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
	all1_b.setSize(1,batch_size)->allocateDevice()->initDeviceConstant(1.0f);
}

BaseLayer::~BaseLayer(){}

void BaseLayer::learningForwardPropagate(mtk::MatrixXf &output,const mtk::MatrixXf& input){
	const float one = 1.0f,zero = 0.0f;
	input.copy(z0);
	cublasSgemm(*cublas,CUBLAS_OP_N,CUBLAS_OP_N,
			output_size,batch_size,1,
			&one,
			b1.getDevicePointer(),output_size,
			all1_b.getDevicePointer(),1,
			&zero,
			u1.getDevicePointer(),output_size);
	cublasSgemm(*cublas,CUBLAS_OP_N,CUBLAS_OP_N,
			output_size,batch_size,input_size,
			&one,
			w1.getDevicePointer(),output_size,
			input.getDevicePointer(),input_size,
			&one,
			u1.getDevicePointer(),output_size);
	this->activation(output,u1);
}

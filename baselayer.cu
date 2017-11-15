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
	u.setSize(output_size,1)->allocateDevice()->initDeviceConstant(0.0f);
}

BaseLayer::~BaseLayer(){}

void BaseLayer::learningForwardPropagation(mtk::MatrixXf &output,const mtk::MatrixXf& input){
	const float one = 1.0f,zero = 0.0f;
	//input.copyTo(z0);
	CUBLAS_HANDLE_ERROR(cublasScopy(*cublas,input.getCols()*input.getRows(),
			input.getDevicePointer(),1,
			z0.getDevicePointer(),1));
	CUBLAS_HANDLE_ERROR(cublasSgemm(*cublas,CUBLAS_OP_N,CUBLAS_OP_N,
			output_size,batch_size,1,
			&one,
			b1.getDevicePointer(),output_size,
			all1_b.getDevicePointer(),1,
			&zero,
			u1.getDevicePointer(),output_size));
	CUBLAS_HANDLE_ERROR(cublasSgemm(*cublas,CUBLAS_OP_N,CUBLAS_OP_N,
			output_size,batch_size,input_size,
			&one,
			w1.getDevicePointer(),output_size,
			input.getDevicePointer(),input_size,
			&one,
			u1.getDevicePointer(),output_size));
	this->activation(output,u1);
}

void BaseLayer::testForwardPropagation(mtk::MatrixXf &output,const mtk::MatrixXf &input) {
	const float one = 1.0f;
	//b1.copyTo(u);
	CUBLAS_HANDLE_ERROR(cublasScopy(*cublas,b1.getCols()*b1.getRows(),
			b1.getDevicePointer(),1,
			u.getDevicePointer(),1));
	CUBLAS_HANDLE_ERROR(cublasSgemm(*cublas,CUBLAS_OP_N,CUBLAS_OP_N,
			output_size,1,input_size,
			&one,
			w1.getDevicePointer(),output_size,
			input.getDevicePointer(),1,
			&one,
			u.getDevicePointer(),output_size));
	this->activation(output,u);
}

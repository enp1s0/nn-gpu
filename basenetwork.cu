/*
 * BaseNetwork
 * ネットワークの親クラス
 * 
 * 2017.11.20
 * mutsuki
 */
#include "basenetwork.h"
#include "matrix_function.h"
#include "hyperparameter.h"
#include "cuda_common.h"
#include <iostream>

using namespace mtk;
template<class T>
__global__ void deviceMap(float *device_ptr_dst,float* device_ptr_src,float a,int max_t){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(max_t <= tid)
		return;
	device_ptr_dst[tid] = T(a)(device_ptr_src[tid]);
}

class AdagradMake{
	float s;
public:
	__device__ AdagradMake(float s):s(s){}
	__device__ float operator()(float x){
		return 1.0f/(sqrtf(x)+s);
	}
};
class MaxAndInverse{
	float m;
public:
	__device__ MaxAndInverse(float m):m(m){}
	__device__ float operator() (float l) const{
		return 1.0f/fmaxf(fabsf(m),fabsf(l));
	}
};

BaseNetwork::BaseNetwork(int input_size,int output_size,int batch_size,std::string network_name,cublasHandle_t cublas,float learning_rate,float adagrad_epsilon,float attenuation_rate):
	input_size(input_size),output_size(output_size),batch_size(batch_size),network_name(network_name),cublas(cublas),learning_rate(learning_rate),adagrad_epsilon(adagrad_epsilon),attenuation_rate(attenuation_rate)
{
	w1.setSize(output_size,input_size)->allocateDevice()->initDeviceRandom(-1.0f,1.0f);
	dw1.setSize(output_size,input_size)->allocateDevice()->initDeviceConstant(0.0f);
	rdw1.setSize(output_size,input_size)->allocateDevice()->initDeviceConstant(0.0f);
	b1.setSize(output_size,1)->allocateDevice()->initDeviceRandom(-1.0f,1.0f);
	db1.setSize(output_size,1)->allocateDevice()->initDeviceConstant(0.0f);
	rdb1.setSize(output_size,1)->allocateDevice()->initDeviceConstant(0.0f);
	u1.setSize(output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	z0.setSize(input_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	adagrad_w1.setSize(output_size,input_size)->allocateDevice()->initDeviceConstant(0.0f);
	adagrad_b1.setSize(output_size,1)->allocateDevice()->initDeviceConstant(0.0f);
	all1_b.setSize(1,batch_size)->allocateDevice()->initDeviceConstant(1.0f);
	all1_o.setSize(1,output_size)->allocateDevice()->initDeviceConstant(1.0f);
	all1_i.setSize(1,input_size)->allocateDevice()->initDeviceConstant(1.0f);
	u.setSize(output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	max_b_i.setSize(output_size,1)->allocateDevice()->initDeviceConstant(0.0f);
	max_w_i.setSize(output_size,input_size)->allocateDevice()->initDeviceConstant(0.0f);
	b1_tmp.setSize(output_size,1)->allocateDevice()->initDeviceConstant(0.0f);
	w1_tmp.setSize(output_size,input_size)->allocateDevice()->initDeviceConstant(0.0f);
	std::cout<<network_name<<"("<<input_size<<","<<output_size<<","<<batch_size<<")"<<std::endl;
	std::cout<<" - learning rate = "<<learning_rate<<std::endl;
	std::cout<<" - adagrad epsilon = "<<adagrad_epsilon<<std::endl;
	std::cout<<" - momentum rate = "<<attenuation_rate<<std::endl;
}

BaseNetwork::~BaseNetwork(){}

void BaseNetwork::learningForwardPropagation(mtk::MatrixXf &output,const mtk::MatrixXf& input){
	const float one = 1.0f,zero = 0.0f;
	mtk::MatrixFunction::copy(cublas, z0, input);
	CUBLAS_HANDLE_ERROR(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
			output_size,batch_size,1,
			&one,
			b1.getDevicePointer(),b1.getRows(),
			all1_b.getDevicePointer(),1,
			&zero,
			u1.getDevicePointer(),u1.getRows()));
	CUBLAS_HANDLE_ERROR(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
			output_size,batch_size,input_size,
			&one,
			w1.getDevicePointer(),w1.getRows(),
			input.getDevicePointer(),input.getRows(),
			&one,
			u1.getDevicePointer(),output_size));
	this->learningActivation(output,u1);
}


void BaseNetwork::learningReflect(){
	const float one = 1.0f;
	const float minus_learning_rate = -learning_rate;
	mtk::MatrixFunction::elementwiseProduct(cublas,adagrad_w1,rdw1,rdw1,1.0f,1.0f);
	mtk::MatrixFunction::elementwiseProduct(cublas,adagrad_b1,rdb1,rdb1,1.0f,1.0f);
	// dw1を作る
	mtk::MatrixFunction::map<AdagradMake>(w1_tmp,adagrad_w1,adagrad_epsilon);
	mtk::MatrixFunction::elementwiseProduct(cublas,dw1,rdw1,w1_tmp,minus_learning_rate,attenuation_rate);
	// db1を作る
	mtk::MatrixFunction::map<AdagradMake>(b1_tmp,adagrad_b1,adagrad_epsilon);
	mtk::MatrixFunction::elementwiseProduct(cublas,db1,rdb1,b1_tmp,minus_learning_rate,attenuation_rate);

	// 更新
	CUBLAS_HANDLE_ERROR( cublasSaxpy( cublas, w1.getSize(),
				&one,
				dw1.getDevicePointer(),1,
				w1.getDevicePointer(),1) );
	CUBLAS_HANDLE_ERROR( cublasSaxpy( cublas, b1.getSize(),
				&one,
				db1.getDevicePointer(),1,
				b1.getDevicePointer(),1) );
	
	// 重みが大きくなりすぎないように
	int max_w_index = 0;
	float zero = 0.0f;
	// 絶対値が最大の要素のindexを返す
	CUBLAS_HANDLE_ERROR( cublasIsamax( cublas,w1.getSize(),
				w1.getDevicePointer(),1,&max_w_index) );
	CUBLAS_HANDLE_ERROR( cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
				1,output_size,1,
				&one,
				w1.getDevicePointer()+max_w_index,1,
				all1_o.getDevicePointer(),1,
				&zero,
				max_b_i.getDevicePointer(),1));
	mtk::MatrixFunction::map<MaxAndInverse>(max_b_i,max_b_i,1.0f);
	CUBLAS_HANDLE_ERROR( cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
				output_size,input_size,1,
				&one,
				max_b_i.getDevicePointer(),output_size,
				all1_i.getDevicePointer(),1,
				&zero,
				max_w_i.getDevicePointer(),max_w_i.getRows()));
	// 正規化
	mtk::MatrixFunction::elementwiseProduct(cublas,w1_tmp,max_w_i,w1);
	mtk::MatrixFunction::elementwiseProduct(cublas,b1_tmp,max_b_i,b1);
	// 結果をコピー
	mtk::MatrixFunction::copy(cublas,w1,w1_tmp);
	mtk::MatrixFunction::copy(cublas,b1,b1_tmp);
}

void BaseNetwork::learningBackPropagation(mtk::MatrixXf &next_error, const mtk::MatrixXf &d2){
	mtk::MatrixFunction::copy(cublas,next_error,d2);
	mtk::MatrixFunction::copy(cublas,d1,d2);
}

mtk::MatrixXf* BaseNetwork::getWeightPointer(){return &w1;}
mtk::MatrixXf* BaseNetwork::getBiasPointer(){return &b1;}
int BaseNetwork::getInputSize(){return input_size;}
int BaseNetwork::getOutputSize(){return output_size;}
std::string BaseNetwork::getNetworkName(){return network_name;}


// test methods

void BaseNetwork::testForwardPropagation(mtk::MatrixXf &output,const mtk::MatrixXf &input) {
	const float one = 1.0f,zero = 0.0f;
	CUBLAS_HANDLE_ERROR(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
			output_size,test_batch_size,1,
			&one,
			b1.getDevicePointer(),b1.getRows(),
			all1_t.getDevicePointer(),1,
			&zero,
			test_u.getDevicePointer(),test_u.getRows()));
	CUBLAS_HANDLE_ERROR(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
			output_size,test_batch_size,input_size,
			&one,
			w1.getDevicePointer(),w1.getRows(),
			input.getDevicePointer(),input.getRows(),
			&one,
			test_u.getDevicePointer(),test_u.getRows()));

	this->testActivation(output,test_u);
}

void BaseNetwork::testInit(int b){
	test_batch_size = b;
	test_u.setSize(output_size,test_batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	all1_t.setSize(1,test_batch_size)->allocateDevice()->initDeviceConstant(1.0f);
}

void BaseNetwork::testRelease(){
	test_u.releaseDevice();
	all1_t.releaseDevice();
}


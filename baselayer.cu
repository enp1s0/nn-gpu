#include "baselayer.h"
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
		return 1.0f/fmaxf(m,l);
	}
};

BaseLayer::BaseLayer(int input_size,int output_size,int batch_size,std::string layer_name,cublasHandle_t cublas,float learning_rate,float adagrad_epsilon,float attenuation_rate):
	input_size(input_size),output_size(output_size),batch_size(batch_size),layer_name(layer_name),cublas(cublas),learning_rate(learning_rate),adagrad_epsilon(adagrad_epsilon),attenuation_rate(attenuation_rate)
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
	u.setSize(output_size,1)->allocateDevice()->initDeviceConstant(0.0f);
	max_b_i.setSize(output_size,1)->allocateDevice()->initDeviceConstant(1.0f);
	max_w_i.setSize(output_size,input_size)->allocateDevice()->initDeviceConstant(1.0f);
	b1_tmp.setSize(output_size,1)->allocateDevice()->initDeviceConstant(0.0f);
	w1_tmp.setSize(output_size,input_size)->allocateDevice()->initDeviceConstant(0.0f);
	std::cout<<layer_name<<"("<<input_size<<","<<output_size<<","<<batch_size<<")"<<std::endl;
	//w1.allocateHost()->copyToHost();
	//w1.print();
}

BaseLayer::~BaseLayer(){}

void BaseLayer::learningForwardPropagation(mtk::MatrixXf &output,const mtk::MatrixXf& input){
	const float one = 1.0f,zero = 0.0f;
	mtk::MatrixFunction::copy(cublas, z0, input);
	CUBLAS_HANDLE_ERROR(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
			output_size,batch_size,1,
			&one,
			b1.getDevicePointer(),output_size,
			all1_b.getDevicePointer(),1,
			&zero,
			u1.getDevicePointer(),output_size));
	CUBLAS_HANDLE_ERROR(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
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
	CUBLAS_HANDLE_ERROR(cublasScopy(cublas,b1.getCols()*b1.getRows(),
			b1.getDevicePointer(),1,
			u.getDevicePointer(),1));
	CUBLAS_HANDLE_ERROR(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
			output_size,1,input_size,
			&one,
			w1.getDevicePointer(),output_size,
			input.getDevicePointer(),1,
			&one,
			u.getDevicePointer(),output_size));
	this->activation(output,u);
}

void BaseLayer::learningReflect(){
	const float one = 1.0f,zero = 0.0f;
	const float minus_learning_rate = -learning_rate;
	CUBLAS_HANDLE_ERROR( cublasSsbmv(cublas,CUBLAS_FILL_MODE_LOWER,
				rdw1.getSize(),0,&one,
				rdw1.getDevicePointer(),1,
				rdw1.getDevicePointer(),1,
				&one,
				adagrad_w1.getDevicePointer(),1) );
	CUBLAS_HANDLE_ERROR( cublasSsbmv(cublas,CUBLAS_FILL_MODE_LOWER,
				rdb1.getSize(),0,&one,
				rdb1.getDevicePointer(),1,
				rdb1.getDevicePointer(),1,
				&one,
				adagrad_b1.getDevicePointer(),1) );
	// dw1を作る
	//deviceMap<AdagradMake><<<BLOCKS,threads_ceildiv(adagrad_w1.getSize(),BLOCKS)>>>(adagrad_w1.getDevicePointer(),adagrad_w1.getDevicePointer(),adagrad_epsilon,adagrad_w1.getSize());
	mtk::MatrixFunction::map<AdagradMake>(adagrad_w1,adagrad_w1,adagrad_epsilon);
	CUBLAS_HANDLE_ERROR( cublasSsbmv(cublas,CUBLAS_FILL_MODE_LOWER,
				rdw1.getSize(),0,
				&minus_learning_rate,
				rdw1.getDevicePointer(),1,
				adagrad_w1.getDevicePointer(),1,
				&attenuation_rate,
				dw1.getDevicePointer(),1));
	// db1を作る
	//deviceMap<AdagradMake><<<BLOCKS,threads_ceildiv(adagrad_b1.getSize(),BLOCKS)>>>(adagrad_w1.getDevicePointer(),adagrad_w1.getDevicePointer(),adagrad_epsilon,adagrad_b1.getSize());
	mtk::MatrixFunction::map<AdagradMake>(adagrad_b1,adagrad_b1,adagrad_epsilon);
	CUBLAS_HANDLE_ERROR( cublasSsbmv(cublas,CUBLAS_FILL_MODE_LOWER,
				rdb1.getSize(),0,
				&minus_learning_rate,
				rdb1.getDevicePointer(),1,
				adagrad_b1.getDevicePointer(),1,
				&attenuation_rate,
				db1.getDevicePointer(),1));

	// 更新
	CUBLAS_HANDLE_ERROR( cublasSaxpy( cublas, w1.getRows() * w1.getCols(),
				&one,
				dw1.getDevicePointer(),1,
				w1.getDevicePointer(),1) );
	CUBLAS_HANDLE_ERROR( cublasSaxpy( cublas, b1.getRows() * b1.getCols(),
				&one,
				db1.getDevicePointer(),1,
				b1.getDevicePointer(),1) );

	// 重みが大きくなりすぎないように
	int max_w_index = 0;
	CUBLAS_HANDLE_ERROR( cublasIsamax( cublas,w1.getRows()*w1.getCols(),
				w1.getDevicePointer(),1,&max_w_index) );
	CUBLAS_HANDLE_ERROR( cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
				1,output_size,1,
				&one,
				w1.getDevicePointer()+max_w_index,1,
				all1_o.getDevicePointer(),1,
				&zero,
				max_b_i.getDevicePointer(),1));
	//deviceMap<MaxAndInverse><<<BLOCKS,threads_ceildiv(max_b_i.getSize(),BLOCKS)>>>(max_b_i.getDevicePointer(),max_b_i.getDevicePointer(),1.0f,max_b_i.getSize());
	mtk::MatrixFunction::map<MaxAndInverse>(max_b_i,max_b_i,1.0f);
	CUBLAS_HANDLE_ERROR( cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
				output_size,input_size,1,
				&one,
				max_b_i.getDevicePointer(),output_size,
				all1_i.getDevicePointer(),input_size,
				&zero,
				max_w_i.getDevicePointer(),output_size));
	// 正規化
	CUBLAS_HANDLE_ERROR( cublasSsbmv(cublas, CUBLAS_FILL_MODE_LOWER,
				max_w_i.getCols() * max_w_i.getRows(),0,
				&one,
				max_w_i.getDevicePointer(),1,
				w1.getDevicePointer(),1,
				&zero,
				w1_tmp.getDevicePointer(),1));
	CUBLAS_HANDLE_ERROR( cublasSsbmv(cublas, CUBLAS_FILL_MODE_LOWER,
				max_b_i.getCols() * max_b_i.getRows(),0,
				&one,
				max_b_i.getDevicePointer(),1,
				b1.getDevicePointer(),1,
				&zero,
				b1_tmp.getDevicePointer(),1));


	// 結果をコピー
	CUBLAS_HANDLE_ERROR( cublasScopy( cublas, w1.getRows()*w1.getCols(),
				w1_tmp.getDevicePointer(),1,
				w1.getDevicePointer(),1) );
	CUBLAS_HANDLE_ERROR( cublasScopy( cublas, b1.getRows()*b1.getCols(),
				b1_tmp.getDevicePointer(),1,
				b1.getDevicePointer(),1) );
}

mtk::MatrixXf* BaseLayer::getWeightPointer(){return &w1;}
mtk::MatrixXf* BaseLayer::getBiasPointer(){return &b1;}

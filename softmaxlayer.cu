#include "softmaxlayer.h"
#include "matrix_function.h"
#include "cuda_common.h"

using namespace mtk;

template<class T>
__global__ void deviceMap(float *device_ptr_dst,float* device_ptr_src,int max_t){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(max_t <= tid)
		return;
	device_ptr_dst[tid] = T()(device_ptr_src[tid]);
}

class Exp{
public:
	__device__ float operator()(float a) const{
		return expf(a);
	}
};
class Inverse{
public:
	__device__ float operator()(float a) const{
		return 1.0f/a;
	}
};

SoftmaxLayer::SoftmaxLayer(int input_size,int output_size,int batch_size,std::string layer_name,cublasHandle_t cublas,float learning_rate,float adagrad_epsilon,float attenuation_rate):
	BaseLayer(input_size,output_size,batch_size,layer_name,cublas,learning_rate,adagrad_epsilon,attenuation_rate)
{
	input_row_0.setSize(1,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	inverse.setSize(output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	output0.setSize(output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
}

SoftmaxLayer::~SoftmaxLayer(){}

void SoftmaxLayer::learningBackPropagation(mtk::MatrixXf& next_error,const mtk::MatrixXf& d2,const mtk::MatrixXf *w2){
	mtk::MatrixFunction::copy(cublas,next_error,d2);
	mtk::MatrixFunction::copy(cublas,d1,d2);
}

void SoftmaxLayer::activation(mtk::MatrixXf& output,const mtk::MatrixXf& input){
	//input行列の0行目を取り出す
	const float one = 1.0f,minus_one = -1.0f,zero = 0.0f;
	mtk::MatrixFunction::copy(cublas,output,input);
	/*CUBLAS_HANDLE_ERROR( cublasScopy(cublas,output.getRows()*output.getCols(),
				input.getDevicePointer(),1,
				output.getDevicePointer(),1) );*/
	// 全列の要素からその列の先頭要素の値を引く
	CUBLAS_HANDLE_ERROR( cublasScopy(cublas,batch_size,
				input.getDevicePointer(), output_size,
				input_row_0.getDevicePointer(),1));
	CUBLAS_HANDLE_ERROR( cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
				output_size,batch_size,1,
				&minus_one,
				all1_o.getDevicePointer(),output_size,
				input_row_0.getDevicePointer(),1,
				&one,
				output.getDevicePointer(),output_size) );
	mtk::MatrixFunction::map<Exp>(output,output);
	// 和を取る
	CUBLAS_HANDLE_ERROR( cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
				1,batch_size,output_size,
				&one,
				all1_o.getDevicePointer(),1,
				output.getDevicePointer(),output_size,
				&zero,
				input_row_0.getDevicePointer(),1));
	// 逆数を計算
	//deviceMap<Inverse><<<BLOCKS,threads_ceildiv(input_row_0.getSize(),BLOCKS)>>>(input_row_0.getDevicePointer(),input_row_0.getDevicePointer(),input_row_0.getSize());
	mtk::MatrixFunction::map<Inverse>(input_row_0,input_row_0);
	// 逆数の行列を計算
	CUBLAS_HANDLE_ERROR( cublasSgemm( cublas, CUBLAS_OP_N,CUBLAS_OP_N,
				output_size,batch_size,1,
				&one,
				all1_o.getDevicePointer(),output_size,
				input_row_0.getDevicePointer(),1,
				&zero,
				inverse.getDevicePointer(),output_size) );
	mtk::MatrixFunction::elementwiseProduct(cublas,output0,output,inverse);
	/*CUBLAS_HANDLE_ERROR(cublasSsbmv(cublas,CUBLAS_FILL_MODE_LOWER,
			inverse.getCols()*inverse.getRows(),0,&one,
			inverse.getDevicePointer(),1,
			output.getDevicePointer(),1,
			&zero,output0.getDevicePointer(),1));*/
	mtk::MatrixFunction::copy(cublas,output,output0);
	/*CUBLAS_HANDLE_ERROR( cublasScopy(cublas,output0.getRows()*output0.getCols(),
				output0.getDevicePointer(),1,
				output.getDevicePointer(),1) );*/
	//output.copyToHost();
	//output.print("output");
	//input.print("input");
}

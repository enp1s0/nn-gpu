#include <iostream>
#include "cuda_common.h"
#include "hiddenlayer.h"
#include "matrix_array.h"
#include "softmaxlayer.h"

const int input_size = 28 * 28;
const int layer0_output_size = 20 * 20;
const int layer1_output_size = 10;
const int batch_size = 64;
const int calc = 20000;
const int test_interval = 500;

int main(){
	cublasHandle_t cublas;
	CUBLAS_HANDLE_ERROR(cublasCreate(&cublas));

	// layers
	mtk::HiddenLayer layer0(input_size,layer0_output_size,batch_size,"layer0",cublas);
	mtk::SoftmaxLayer layer1(layer0_output_size,layer1_output_size,batch_size,"layer1",cublas);

	// feature
	mtk::MatrixXf input,hidden0,output;
	input.setSize(input_size,batch_size)->allocateDevice()->initDeviceRandom(-1.0f,1.0f);
	hidden0.setSize(layer0_output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	output.setSize(layer1_output_size,batch_size)->allocateDevice()->allocateHost()->initDeviceConstant(0.0f);

	for(int c = 0;c < calc;c++){
		// 順方向計算
		layer0.learningForwardPropagation(hidden0,input);
		layer1.learningForwardPropagation(output,hidden0);
		// 逆方向計算

		// 反映
		layer0.learningReflect();
		layer1.learningReflect();
		if((c+1)%test_interval == 0){std::cout<<c<<std::endl;}
	}
	CUBLAS_HANDLE_ERROR(cublasDestroy( cublas));
}

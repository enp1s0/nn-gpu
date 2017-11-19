#include <iostream>
#include "cuda_common.h"
#include "hiddenlayer.h"
#include "matrix_array.h"
#include "softmaxlayer.h"
#include "matrix_function.h"
#include "mnist.h"

const int input_size = 28 * 28;
const int layer0_output_size = 15 * 15;
const int layer1_output_size = 10;
const int batch_size = 32;
const int calc = 10000;
const int test_interval = 1000;

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

	// error 
	mtk::MatrixXf input_error,hidden0_error,output_error;
	input_error.setSize(layer0_output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	hidden0_error.setSize(layer1_output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	output_error.setSize(layer1_output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);

	// teacher
	mtk::MatrixXf teacher;
	teacher.setSize(layer1_output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	// 学習データ
	std::cout<<"Loading training data ... ";std::cout.flush();
	mtk::MNISTLoader mnist;
	if(mnist.loadMNISTTrainData("train-images-idx3-ubyte","train-labels-idx1-ubyte")){
		std::cout<<std::endl;
		std::cerr<<"invalid training file name"<<std::endl;
		return 1;
	}
	std::cout<<"DONE"<<std::endl;
	std::cout<<"Start training"<<std::endl;
	float minus_one = -1.0f;
	for(int c = 0;c < calc;c++){
		mnist.setTrainDataToMatrix(input,teacher,batch_size);
		// 順方向計算
		layer0.learningForwardPropagation(hidden0,input);
		layer1.learningForwardPropagation(output,hidden0);
		// 出力層の誤差計算
		mtk::MatrixFunction::copy(cublas,output_error,output);
		CUBLAS_HANDLE_ERROR(cublasSaxpy(cublas,output.getSize(), &minus_one,
					teacher.getDevicePointer(),1,
					output_error.getDevicePointer(),1));
		// 逆方向計算
		layer1.learningBackPropagation(	hidden0_error, output_error);
		layer0.learningBackPropagation( input_error, hidden0_error, layer1.getWeightPointer());
		// 反映
		layer0.learningReflect();
		layer1.learningReflect();
		if((c+1)%test_interval == 0){std::cout<<(c+1)<<" / "<<calc<<" ("<<(100.0f*(c+1)/calc)<<"%)"<<std::endl;}
	}
	//hidden0.allocateHost()->copyToHost()->print("hidden");
	//output.copyToHost()->print("output");
	//output_error.allocateHost()->copyToHost()->print("output error");
	std::cout<<"Done"<<std::endl;
	CUBLAS_HANDLE_ERROR(cublasDestroy( cublas));
}

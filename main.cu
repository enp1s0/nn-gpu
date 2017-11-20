#include <iostream>
#include "cuda_common.h"
#include "hiddennetwork.h"
#include "matrix_array.h"
#include "softmaxnetwork.h"
#include "matrix_function.h"
#include "mnist.h"
#include "cuda_event.h"

const int input_size = 28 * 28;
const int network0_output_size = 15 * 15;
const int network1_output_size = 10;
const int batch_size = 64;
const int calc = 10000;
const int test_interval = 100;

int main(){
	mtk::CudaEvent event;
	event.createEvent("init_start")
		->createEvent("init_done")
		->createEvent("calc_start")
		->createEvent("calc_done");
	event.recordEvent("init_start");
	cublasHandle_t cublas;
	CUBLAS_HANDLE_ERROR(cublasCreate(&cublas));

	// networks
	mtk::HiddenNetwork network0(input_size,network0_output_size,batch_size,"network0",cublas);
	mtk::SoftmaxNetwork network1(network0_output_size,network1_output_size,batch_size,"network1",cublas);

	// feature
	mtk::MatrixXf input,hidden0,output;
	input.setSize(input_size,batch_size)->allocateDevice()->initDeviceRandom(-1.0f,1.0f);
	hidden0.setSize(network0_output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	output.setSize(network1_output_size,batch_size)->allocateDevice()->allocateHost()->initDeviceConstant(0.0f);

	// error 
	mtk::MatrixXf input_error,hidden0_error,output_error;
	input_error.setSize(network0_output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	hidden0_error.setSize(network1_output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	output_error.setSize(network1_output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);

	// teacher
	mtk::MatrixXf teacher;
	teacher.setSize(network1_output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	// 学習データ
	std::cout<<"Loading training data ... ";std::cout.flush();
	mtk::MNISTLoader mnist;
	if(mnist.loadMNISTTrainData("train-images-idx3-ubyte","train-labels-idx1-ubyte")){
		std::cout<<std::endl;
		std::cerr<<"invalid training file name"<<std::endl;
		return 1;
	}
	event.recordEvent("init_done");
	event.recordEvent("calc_start");
	std::cout<<"DONE : "<<event.elapsedTime("init_start","init_done")<<" [ms]"<<std::endl; 
	std::cout<<"Start training"<<std::endl;
	float minus_one = -1.0f;
	for(int c = 0;c < calc;c++){
		mnist.setTrainDataToMatrix(input,teacher,batch_size);
		// 順方向計算
		network0.learningForwardPropagation(hidden0,input);
		network1.learningForwardPropagation(output,hidden0);
		// 出力層の誤差計算
		mtk::MatrixFunction::copy(cublas,output_error,output);
		CUBLAS_HANDLE_ERROR(cublasSaxpy(cublas,output.getSize(), &minus_one,
					teacher.getDevicePointer(),1,
					output_error.getDevicePointer(),1));
		// 逆方向計算
		network1.learningBackPropagation(	hidden0_error, output_error);
		network0.learningBackPropagation( input_error, hidden0_error, network1.getWeightPointer());
		// 反映
		network0.learningReflect();
		network1.learningReflect();
		if((c+1)%test_interval == 0){std::cout<<(c+1)<<" / "<<calc<<" ("<<(100.0f*(c+1)/calc)<<"%)"<<std::endl;}
	}
	//hidden0.allocateHost()->copyToHost()->print("hidden");
	//output.copyToHost()->print("output");
	output_error.allocateHost()->copyToHost()->print("output error");
	event.recordEvent("calc_done");
	std::cout<<"Done : "<<event.elapsedTime("calc_start","calc_done")<<" [ms]"<<std::endl; 
	CUBLAS_HANDLE_ERROR(cublasDestroy( cublas));
}

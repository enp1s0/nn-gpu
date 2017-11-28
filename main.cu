#include <iostream>
#include "cuda_common.h"
#include "hiddennetwork.h"
#include "matrix_array.h"
#include "softmaxnetwork.h"
#include "matrix_function.h"
#include "mnist.h"
#include "cuda_event.h"
#include "neuralnetwork.h"
#include "aggregation.h"

const int input_size = 28 * 28;
const int network0_output_size = 5 * 10;
//const int network1_output_size = 15 * 15;
const int last_output_size = 10;
const int batch_size = 32;
const int calc = 100000;
const int test_interval = 10000;

const int test_batch_size = batch_size;


int main(){
	mtk::CudaEvent event;
	event.createEvent("init_start")
		->createEvent("init_done")
		->createEvent("calc_start")
		->createEvent("calc_done");
	event.recordEvent("init_start");
	cublasHandle_t cublas;
	CUBLAS_HANDLE_ERROR(cublasCreate(&cublas));

	mtk::NeuralNetwork network(batch_size,cublas);
	network.add(new mtk::HiddenUnit(input_size,network0_output_size,batch_size,"first network",cublas,1.2,1.0f,0.5f))
		//->add(new mtk::HiddenUnit(network0_output_size,network1_output_size,batch_size,"second network",cublas))
		->add(new mtk::SoftmaxUnit(network0_output_size,last_output_size,batch_size,"last network",cublas))
		->construct();
	mtk::MatrixXf input,teacher,error,output;
	input.setSize(input_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	output.setSize(last_output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	teacher.setSize(last_output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	error.setSize(last_output_size,batch_size)->allocateDevice()->initDeviceConstant(0.0f);

	mtk::Aggregation aggregation(test_batch_size,10,cublas);
	network.testInit(test_batch_size);
	mtk::MatrixXf test_input,test_output,test_teacher;
	test_input.setSize(input_size,test_batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	test_output.setSize(last_output_size,test_batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	test_teacher.setSize(last_output_size,test_batch_size)->allocateDevice()->initDeviceConstant(0.0f);

	// 学習データ
	std::cout<<"Loading training data ... ";std::cout.flush();
	mtk::MNISTLoader mnist;
	if(mnist.loadMNISTTrainData("train-images-idx3-ubyte","train-labels-idx1-ubyte")){
		std::cout<<std::endl;
		std::cerr<<"invalid training file name"<<std::endl;
		return 1;
	}
	if(mnist.loadMNISTTestData("t10k-images-idx3-ubyte","t10k-labels-idx1-ubyte")){
		std::cout<<std::endl;
		std::cerr<<"invalid training file name"<<std::endl;
		return 1;
	}
	event.recordEvent("init_done");
	event.recordEvent("calc_start");
	std::cout<<"DONE : "<<event.elapsedTime("init_start","init_done")<<" [ms]"<<std::endl; 
	std::cout<<"Start training"<<std::endl;
	for(int c = 0;c < calc;c++){
		mnist.setTrainDataToMatrix(input,teacher,batch_size);
		network.learningForwardPropagation(output,input)->calcError(error,output,teacher)->learningBackPropagation(error);
		if((c+1)%test_interval == 0){
			aggregation.accuracyClear();
			for(int i = 0;i < 10000;i+=test_batch_size){
				mnist.setTestDataToMatrix(test_input, test_teacher,i,test_batch_size);
				network.testForwardPropagation(test_output,test_input);
				aggregation.accuracyCompareWithTeacher(test_output,test_teacher);
			}
			std::cout<<(c+1)<<" / "<<calc<<" ("<<(100.0f*(c+1)/calc)<<"%)"<<std::endl;
			std::cout<<" - accuracy = "<<aggregation.accuracyCalcAccuracy()*100<<" %"<<std::endl;
		}
	}
	mtk::MatrixXf result_matrix;
	result_matrix.setSize(last_output_size,last_output_size)->allocateDevice()->allocateHost()->initDeviceConstant(0.0f);
	for(int i = 0;i < 10000;i+=test_batch_size){
		mnist.setTestDataToMatrix(test_input, test_teacher,i,test_batch_size);
		network.testForwardPropagation(test_output,test_input);
		aggregation.matrixCompareWithTeacher(result_matrix,test_output,test_teacher);
	}
	result_matrix.copyToHost()->print("result_matrix");

	//aggregation.clear();
	//aggregation.compareWithTeacher(output,teacher);
	//std::cout<<" - accuracy = "<<aggregation.calcAccuracy()*100<<" %"<<std::endl;
	//output.allocateHost()->copyToHost()->print("output");
	//error.allocateHost()->copyToHost()->print("output error");
	//teacher.allocateHost()->copyToHost()->print("teacher");
	event.recordEvent("calc_done");
	std::cout<<"Done : "<<event.elapsedTime("calc_start","calc_done")<<" [ms]"<<std::endl; 
	CUBLAS_HANDLE_ERROR(cublasDestroy( cublas));
}

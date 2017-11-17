#include "mnist.h"
#include "cuda_common.h"
#include <curand.h>
#include <curand_kernel.h>
#include <random>

using namespace mtk;

__global__ void deviceRandomArrangement(float* input,float* teacher,float *image_data,float *label_data,int max_t){
	
}

MNISTLoader::MNISTLoader(){}
int MNISTLoader::reverse(int n){
	char a0,a1,a2,a3;
	a0 = (n>>24) & 255;
	a1 = (n>>16) & 255;
	a2 = (n>>8) & 255;
	a3 = n & 255;
	return ((int)a3 << 24) + ((int)a2 << 16) + ((int)a1 << 8) + a0;
}

void printImage(float *data){
	for(int i = 0;i < 28;i++){
		for(int j = 0;j < 28;j++){
			printf("%02d",(int)(data[i*28+j]*99));
		}
		printf("\n");
	}
}

void MNISTLoader::printTestImage(int n){
	printImage( test_data_vector[n]->data );
}

void MNISTLoader::setTrainDataToMatrix(mtk::MatrixXf& input,mtk::MatrixXf& teacher,int batch_size){
	teacher.initDeviceConstant(0.0f);
}

int MNISTLoader::setTestDataToMatrix(mtk::MatrixXf& input,int index){
	MNISTData* data = test_data_vector[index];
	for(int j = 0;j < 28*28;j++){
		//input(j) = data->data[j];
	}
	return data->label;
}

int MNISTLoader::loadMNISTData(std::string image_filename,std::string label_filename,std::vector<MNISTData*>& data_vector){
	std::ifstream image_ifs(image_filename,std::ios::binary);
	std::ifstream label_ifs(label_filename,std::ios::binary);
	if(!image_ifs|| !label_ifs ){
		return 1;
	}

	int magic_number,amount,row,col;
	int label;
	char read_1byte;
	int read_1byte_int;
	image_ifs.read((char*)&magic_number,sizeof(magic_number));
	magic_number = reverse( magic_number );
	image_ifs.read((char*)&amount,sizeof(amount));
	amount = reverse( amount );
	image_ifs.read((char*)&row,sizeof(row));
	row = reverse( row );
	image_ifs.read((char*)&col,sizeof(col));
	col = reverse( col );
	label_ifs.read((char*)&magic_number,sizeof(magic_number));
	magic_number = reverse( magic_number );
	label_ifs.read((char*)&amount,sizeof(amount));
	amount = reverse( amount );
	//std::cout<<"magic_number = "<<(int)magic_number<<std::endl<<"row = "<<(int)row<<std::endl<<"col = " <<(int)col<<std::endl<<"amount = "<<(int)amount<<std::endl;
	for(int a = 0;a < train_data_amount;a++){
		MNISTData *data = new MNISTData;
		label_ifs.read((char*)&label,sizeof(char));
		label &= 0xf;
		data->label = ( label );
		for(int i = 0;i < 28*28;i++){
			image_ifs.read((char*)&read_1byte_int,sizeof(char));
			read_1byte_int &= 0xf;
			//data->data[i] = read_1byte_int/255.0f/28.0f/28.0f;
			data->data[i] = read_1byte_int/255.0f;
		}
		if( label > 9 ){
			std::cout<<"label index = "<< a <<std::endl;
			printImage(data->data);
			return 1;
		}
		data_vector.push_back(data);
	}
	image_ifs.close();
	label_ifs.close();
	return 0;
}

int MNISTLoader::loadMNISTTrainData(std::string image_filename,std::string label_filename){
	int res = this->loadMNISTData(image_filename,label_filename,train_data_vector);
	for(int i = 0;i < train_data_amount;i++){
		MNISTData* data = train_data_vector[i];
		for(int j = 0;j < data_dim*data_dim;j++){
			image_data.getHostPointer()[j + i * 28 * 28] = data->data[j];
		}
		label_data.getHostPointer()[data->label + i * 10] = 1.0f;
		//teacher(data->label,i) = 1.0f;
	}
	return res;
}
int MNISTLoader::loadMNISTTestData(std::string image_filename,std::string label_filename){
	return this->loadMNISTData(image_filename,label_filename,test_data_vector);
}

MNISTLoader::~MNISTLoader(){
	for(auto data : train_data_vector){
		delete data;
	}
}

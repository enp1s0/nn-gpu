#include "mnist.h"
#include "cuda_common.h"
#include <curand.h>
#include <curand_kernel.h>
#include <random>

using namespace mtk;

__global__ void deviceRandomArrangement(float* input,float* teacher,float *image_data,float *label_data,int batch_size,int seed){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batch_size)return;
	curandState s;
	curand_init(seed,tid,0,&s);
	int data_id = curand_uniform(&s) * 60000;
	for(int i = 0;i < 28*28;i++){
		input[i+tid*28*28] = image_data[i+data_id*28*28];
	}
	for(int i = 0;i < 10;i++){
		teacher[i + tid*10] = label_data[i + data_id*10];
	}
}

__global__ void devieSequentialArrangement(float* input,float* teacher,float* image_data,float* label_data,int start,int batch_size){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= batch_size)return;
	for(int i = 0;i < 28*28;i++){
		input[i+tid*28*28] = image_data[i+(start+tid)*28*28];
	}
	for(int i = 0;i < 10;i++){
		teacher[i + tid*10] = label_data[i + (start+tid)*10];
	}
}

MNISTLoader::MNISTLoader(){
	std::random_device rnd;
	mt.seed(rnd());
}
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
	deviceRandomArrangement<<<64,threads_ceildiv(batch_size,64)>>>(input.getDevicePointer(),teacher.getDevicePointer(),image_data.getDevicePointer(),label_data.getDevicePointer(),batch_size,mt());
}

void MNISTLoader::setTestDataToMatrix(mtk::MatrixXf& input,mtk::MatrixXf& teacher,int start,int batch_size){
	devieSequentialArrangement<<<64,threads_ceildiv(batch_size,64)>>>(input.getDevicePointer(),teacher.getDevicePointer(),test_image_data.getDevicePointer(),test_label_data.getDevicePointer(),start,batch_size);
}

int MNISTLoader::loadMNISTData(std::string image_filename,std::string label_filename,std::vector<MNISTData*>& data_vector){
	std::ifstream image_ifs(image_filename,std::ios::binary);
	std::ifstream label_ifs(label_filename,std::ios::binary);
	if(!image_ifs|| !label_ifs ){
		return 1;
	}

	int magic_number,amount,row,col;
	int label;
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
	for(int a = 0;a < train_data_amount;a++){
		MNISTData *data = new MNISTData;
		label_ifs.read((char*)&label,sizeof(char));
		label &= 0xf;
		data->label = ( label );
		for(int i = 0;i < 28*28;i++){
			image_ifs.read((char*)&read_1byte_int,sizeof(char));
			read_1byte_int &= 0xf;
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
	image_data.setSize(28*28,train_data_amount)->allocateDevice()->allocateHost()->initDeviceConstant(0.0f);
	label_data.setSize(10,train_data_amount)->allocateDevice()->allocateHost()->initDeviceConstant(0.0f);
	int res = this->loadMNISTData(image_filename,label_filename,train_data_vector);
	if(res != 0)
		return res;
	for(int i = 0;i < train_data_amount;i++){
		MNISTData* data = train_data_vector[i];
		for(int j = 0;j < data_dim*data_dim;j++){
			image_data.getHostPointer()[j + i * 28 * 28] = data->data[j];
		}
		label_data.getHostPointer()[data->label + i * 10] = 1.0f;
	}
	image_data.copyToDevice();
	label_data.copyToDevice();
	return res;
}
int MNISTLoader::loadMNISTTestData(std::string image_filename,std::string label_filename){
	test_image_data.setSize(28*28,test_data_amount)->allocateDevice()->allocateHost()->initDeviceConstant(0.0f);
	test_label_data.setSize(10,test_data_amount)->allocateDevice()->allocateHost()->initDeviceConstant(0.0f);
	int res = this->loadMNISTData(image_filename,label_filename,test_data_vector);
	if(res != 0)
		return res;
	for(int i = 0;i < test_data_amount;i++){
		MNISTData* data = test_data_vector[i];
		for(int j = 0;j < data_dim*data_dim;j++){
			test_image_data.getHostPointer()[j + i * 28 * 28] = data->data[j];
		}
		test_label_data.getHostPointer()[data->label + i * 10] = 1.0f;
	}
	test_image_data.copyToDevice();
	test_label_data.copyToDevice();
	return res;
}

MNISTLoader::~MNISTLoader(){
	for(auto data : train_data_vector){
		delete data;
	}
}

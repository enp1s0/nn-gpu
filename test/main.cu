#include "matrix_array.h"
#include "cublas_common.h"
#include "matrix_functions.h"
#include <iostream>
const int BLOCKS = 1 << 7;

void showMatrix(mtk::MatrixXf &mat0){
	for(int i = 0;i < mat0.getRows();i++){
		for(int j = 0;j < mat0.getCols();j++){
			std::cout<<mat0.getHostPointer()[i+j*mat0.getRows()]<<" ";
		}
		std::cout<<std::endl;
	}
}
template<class T>
__global__ void deviceMap(float *device_ptr_dst,float* device_ptr_src,int max_t){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(max_t <= tid)
		return;
	device_ptr_dst[tid] = T()(device_ptr_src[tid]);
}
class POI{
public:
	__device__ float operator()(float a){
		return a*a;
	}
};

void unary(){
	mtk::MatrixXf mat0,mat1;
	mat0.setSize(10,5)->allocateDevice()->allocateHost()->initDeviceRandom(-1.0f,1.0f);
	mat1.setSize(10,5)->allocateDevice()->allocateHost()->initDeviceConstant(0.0f);

	deviceMap<POI><<<BLOCKS,(mat0.getCols()*mat0.getRows()+BLOCKS-1)/BLOCKS>>>(mat1.getDevicePointer(),mat0.getDevicePointer(),mat0.getCols() * mat0.getRows());

	mat0.copyToHost();
	mat1.copyToHost();

	std::cout<<"mat0"<<std::endl;
	showMatrix(mat0);
	std::cout<<"mat1"<<std::endl;
	showMatrix(mat1);
}
void element_wise(cublasHandle_t cublas){
	mtk::MatrixXf mat0,mat1,mat2;
	mat0.setSize(10,5)->allocateDevice()->allocateHost()->initDeviceRandom(-1.0f,1.0f);
	mat1.setSize(10,5)->allocateDevice()->allocateHost()->initDeviceRandom(-1.0f,1.0f);
	mat2.setSize(10,5)->allocateDevice()->allocateHost()->initDeviceRandom(-1.0f,1.0f);
	/*const float one = 1.0f,zero = 0.0f;
	CUBLAS_HANDLE_ERROR( cublasSsbmv( cublas, CUBLAS_FILL_MODE_LOWER,
				mat0.getRows()*mat0.getCols(),0,&one,
				mat0.getDevicePointer(),1,
				mat1.getDevicePointer(),1,
				&zero,mat2.getDevicePointer(),1));*/
	mtk::CublasFunction::elementwiseProduct(cublas,mat2,mat0,mat1);
	mat0.copyToHost();
	mat1.copyToHost();
	mat2.copyToHost();
	std::cout<<"mat0"<<std::endl;
	showMatrix(mat0);
	std::cout<<"mat1"<<std::endl;
	showMatrix(mat1);
	std::cout<<"mat2"<<std::endl;
	showMatrix(mat2);
}

int main(){
	cublasHandle_t cublas;
	CUBLAS_HANDLE_ERROR(cublasCreate(&cublas));
	element_wise(cublas);
	//unary();
	CUBLAS_HANDLE_ERROR(cublasDestroy( cublas));
}


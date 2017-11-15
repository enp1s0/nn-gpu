#include "matrix_array.h"
#include "cublas_common.h"
#include <iostream>

void showMatrix(mtk::MatrixXf &mat0){
	for(int i = 0;i < mat0.getRows();i++){
		for(int j = 0;j < mat0.getCols();j++){
			std::cout<<mat0.getHostPointer()[i+j*mat0.getRows()]<<" ";
		}
		std::cout<<std::endl;
	}
}

int main(){
	cublasHandle_t cublas;
	CUBLAS_HANDLE_ERROR(cublasCreate(&cublas));
	mtk::MatrixXf mat0,mat1,mat2;
	mat0.setSize(10,5)->allocateDevice()->allocateHost()->initDeviceRandom(-1.0f,1.0f);
	mat1.setSize(10,5)->allocateDevice()->allocateHost()->initDeviceRandom(-1.0f,1.0f);
	mat2.setSize(10,5)->allocateDevice()->allocateHost()->initDeviceRandom(-1.0f,1.0f);
	const float one = 1.0f,zero = 0.0f;
	CUBLAS_HANDLE_ERROR( cublasSsbmv( cublas, CUBLAS_FILL_MODE_LOWER,
				mat0.getRows()*mat0.getCols(),0,&one,
				mat0.getDevicePointer(),1,
				mat1.getDevicePointer(),1,
				&zero,mat2.getDevicePointer(),1));
	mat0.copyToHost();
	mat1.copyToHost();
	mat2.copyToHost();
	std::cout<<"mat0"<<std::endl;
	showMatrix(mat0);
	std::cout<<"mat1"<<std::endl;
	showMatrix(mat1);
	std::cout<<"mat2"<<std::endl;
	showMatrix(mat2);
	CUBLAS_HANDLE_ERROR(cublasDestroy( cublas));
}

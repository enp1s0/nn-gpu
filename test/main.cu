#include "matrix_array.h"
#include <iostream>

int main(){
	mtk::MatrixXf mat;
	mat.setSize(10,10)->allocateDevice()->allocateHost()->initDeviceRandom(-10.0f,1.0f);
	mat.copyToHost();
	for(int i = 0;i < mat.getCols();i++){
		for(int j = 0;j < mat.getRows();j++){
			std::cout<<mat.getHostPointer()[j+i*mat.getRows()]<<" ";
		}
		std::cout<<std::endl;
	}
}

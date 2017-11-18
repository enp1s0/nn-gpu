#pragma once
#include <string>
namespace mtk{
	class MatrixXf{
		int rows,cols;
		float *device_ptr,*host_ptr;
	public:
		MatrixXf();
		MatrixXf(int rows,int cols);
		~MatrixXf();
		MatrixXf* setSize(int rows,int cols);
		MatrixXf* allocateHost();
		MatrixXf* allocateDevice();
		int getCols()const;
		int getRows()const ;
		int getSize() const;
		float* getDevicePointer() const;
		float* getHostPointer() const ;
		MatrixXf* copyToDevice();
		MatrixXf* copyToHost();
		MatrixXf* copyTo(MatrixXf& dest_matrix) ;
		MatrixXf* copyTo(float* dest_ptr) ;
		void operator=(const MatrixXf m);
		MatrixXf* initDeviceRandom(float min,float max);
		MatrixXf* initDeviceConstant(float f);
		MatrixXf* print(std::string label="") ;
	};
}

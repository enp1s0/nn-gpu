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
		void copyToDevice();
		void copyToHost();
		void copyTo(MatrixXf& dest_matrix) const;
		void copyTo(float* dest_ptr) const;
		void operator=(const MatrixXf m);
		void initDeviceRandom(float min,float max);
		void initDeviceConstant(float f);
		void print(std::string label="") const;
	};
}

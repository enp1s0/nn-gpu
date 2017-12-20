#pragma once
#include <string>
namespace mtk{
	class MatrixXf{
		// 連続メモリ確保のために木構造を持つ
		// ただし親を記憶しない
		// depthが0の場合のみデストラクタでメモリの解放を行う
		int depth;
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
		void setDevicePointer(float* device_ptr);
		void setHostPointer(float* host_ptr);
		MatrixXf* copyToDevice();
		MatrixXf* copyToHost();
		MatrixXf* copyTo(MatrixXf& dest_matrix) ;
		MatrixXf* copyTo(float* dest_ptr) ;
		void operator=(const MatrixXf m);
		MatrixXf* initDeviceRandom(float min,float max);
		MatrixXf* initDeviceConstant(float f);
		MatrixXf* print(std::string label="") ;
		MatrixXf* releaseDevice();
		MatrixXf* releaseHost();
		// メモリを2つに割る
		MatrixXf* splitDevice(mtk::MatrixXf& s0,mtk::MatrixXf& s1);
		MatrixXf* splitHost(mtk::MatrixXf& s0,mtk::MatrixXf& s1) ;
	};
}

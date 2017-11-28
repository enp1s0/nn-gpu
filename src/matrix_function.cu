#include "matrix_function.h"

using namespace mtk;

void MatrixFunction::copy(cublasHandle_t cublas, mtk::MatrixXf &dst, const mtk::MatrixXf &src){
	CUBLAS_HANDLE_ERROR( cublasScopy(cublas,dst.getCols() * dst.getRows(),
				src.getDevicePointer(),1,
				dst.getDevicePointer(),1) );
}

void MatrixFunction::elementwiseProduct(cublasHandle_t cublas, mtk::MatrixXf &dst, const mtk::MatrixXf &src0, const mtk::MatrixXf &src1,float a,float b){
	CUBLAS_HANDLE_ERROR( cublasSsbmv( cublas, CUBLAS_FILL_MODE_LOWER,
				src0.getSize(),0,&a,
				src0.getDevicePointer(),1,
				src1.getDevicePointer(),1,
				&b,dst.getDevicePointer(),1) );
}

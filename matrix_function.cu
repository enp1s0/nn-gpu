#include "matrix_function.h"

using namespace mtk;

void MatrixFunction::copy(cublasHandle_t cublas, mtk::MatrixXf &dst, const mtk::MatrixXf &src){
	CUBLAS_HANDLE_ERROR( cublasScopy(cublas,dst.getCols() * dst.getRows(),
				src.getDevicePointer(),1,
				dst.getDevicePointer(),1) );
}

void MatrixFunction::elementwiseProduct(cublasHandle_t cublas, mtk::MatrixXf &dst, const mtk::MatrixXf &src0, const mtk::MatrixXf &src1){
	const float zero = 0.0f,one = 1.0f;
	CUBLAS_HANDLE_ERROR( cublasSsbmv( cublas, CUBLAS_FILL_MODE_LOWER,
				src0.getCols()*src0.getRows(),0,&one,
				src0.getDevicePointer(),1,
				src1.getDevicePointer(),1,
				&zero,dst.getDevicePointer(),1) );
}

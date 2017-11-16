#pragma once

#include "matrix_array.h"
#include "cublas_common.h"
namespace mtk {
	class MatrixFunction{
	public:
		static void copy(cublasHandle_t cublas,mtk::MatrixXf& dst,const mtk::MatrixXf& src);
		static void elementwiseProduct(cublasHandle_t cublas,mtk::MatrixXf& dst,const mtk::MatrixXf& src0,const mtk::MatrixXf& src1);
	};
}

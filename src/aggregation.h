#pragma once
#include "matrix_array.h"

namespace mtk{
	class Aggregation{
		int batch_size,class_size,all_test_size,correct;
		mtk::MatrixXf result;
		mtk::MatrixXf all1_b;
		cublasHandle_t cublas;
	public:
		Aggregation(int class_size,int batch_size,cublasHandle_t cublas);
		void accuracyCompareWithTeacher(const mtk::MatrixXf& output,const mtk::MatrixXf& teacher);
		void accuracyClear();
		float accuracyCalcAccuracy() const;
		void matrixCompareWithTeacher(mtk::MatrixXf& result,const mtk::MatrixXf& output,const mtk::MatrixXf& teacher);
	};
}

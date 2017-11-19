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
		void compareWithTeacher(const mtk::MatrixXf& output,const mtk::MatrixXf& teacher);
		void clear();
		float calcAccuracy() const;
	};
}

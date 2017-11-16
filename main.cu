#include <iostream>
#include "cuda_common.h"
#include "hiddenlayer.h"
#include "matrix_array.h"


int main(){
	cublasHandle_t cublas;
	CUBLAS_HANDLE_ERROR(cublasCreate(&cublas));
	mtk::HiddenLayer h0(14*14,28*28,100,"h",cublas);
	//mtk::HiddenLayer h0(100,100,100,"h",&cublas,10.f,1.0f,1.0f);
	for(int i = 0;i < 100;i++)
		h0.learningReflect();
	CUBLAS_HANDLE_ERROR(cublasDestroy( cublas));
}

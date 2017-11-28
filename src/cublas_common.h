#pragma once
#include <cublas.h>
#include <cublas_v2.h>


#define CUBLAS_HANDLE_ERROR( err ) (cuBlasStatusError( err, __FILE__, __LINE__ ))
const char* cublasGetErrorString(cublasStatus_t status);
void cuBlasStatusError( cublasStatus_t err,const char *file,int line );

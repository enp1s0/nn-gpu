#include <iostream>
#include "cuda_common.h"
#include "cublas_common.h"
#include "aggregation.h"

const int BLOCKS = 1 << 5;
const int THREADS = 1 << 5;

__global__ void deviceCompareWithTeacher(float *result,float *output_ptr ,float *teacher_ptr ,int class_size,int max_t,int loop){
	for(int l = 0;l < loop;l++){
		int tid = (threadIdx.x + blockIdx.x * blockDim.x) * loop + l;
		if(tid >= max_t)
			return;
		float *my_output_ptr = output_ptr + tid * class_size;
		float *my_teacher_ptr = teacher_ptr + tid * class_size;
		int output_max_index = 0,teacher_max_index = 0;
		float output_max = 0.0f;
		for(int i = 0;i < class_size;i++){
			if(my_output_ptr[i] > output_max){
				output_max = my_output_ptr[i];
				output_max_index = i;
			}
			if(my_teacher_ptr[i] > 0.5f){
				teacher_max_index = i;
			}
		}
		if( output_max_index == teacher_max_index )
			result[tid] = 1.0f;
		else
			result[tid] = 0.0f;
	}
}
__global__ void deviceMatrixCompareWithTeacher(float *result,float *output_ptr ,float *teacher_ptr ,int class_size,int max_t,int loop){
	for(int l = 0;l < loop;l++){
		int tid = (threadIdx.x + blockIdx.x * blockDim.x) * loop + l;
		if(tid >= max_t)
			return;
		float *my_output_ptr = output_ptr + tid * class_size;
		float *my_teacher_ptr = teacher_ptr + tid * class_size;
		int output_max_index = 0,teacher_max_index = 0;
		float output_max = 0.0f;
		for(int i = 0;i < class_size;i++){
			if(my_output_ptr[i] > output_max){
				output_max = my_output_ptr[i];
				output_max_index = i;
			}
			if(my_teacher_ptr[i] > 0.5f){
				teacher_max_index = i;
			}
		}
		atomicAdd(result+(output_max_index + teacher_max_index * class_size),1.0f);
	}
}

mtk::Aggregation::Aggregation(int batch_size,int class_size,cublasHandle_t cublas):
	batch_size(batch_size),class_size(class_size),correct(0),all_test_size(0),cublas(cublas)
{
	result.setSize(1,batch_size)->allocateDevice()->initDeviceConstant(0.0f);
	all1_b.setSize(1,batch_size)->allocateDevice()->initDeviceConstant(1.0f);
}

void mtk::Aggregation::accuracyClear(){
	result.initDeviceConstant(0.0f);
	all_test_size = 0;
	correct = 0;
}

void mtk::Aggregation::accuracyCompareWithTeacher(const mtk::MatrixXf& output,const mtk::MatrixXf& teacher){
	float result_v = 0.0f;
	deviceCompareWithTeacher<<<BLOCKS,std::min(THREADS,threads_ceildiv(output.getCols(),BLOCKS))>>>(result.getDevicePointer(),output.getDevicePointer(),teacher.getDevicePointer(),class_size,batch_size,threads_ceildiv(threads_ceildiv(output.getCols(),BLOCKS),THREADS));
	CUBLAS_HANDLE_ERROR( cublasSdot(cublas, result.getCols(),
				result.getDevicePointer(), 1,
				all1_b.getDevicePointer(), 1,
				&result_v ) );
	CUDA_HANDLE_ERROR( cudaDeviceSynchronize() );
	correct += (int)result_v;
	all_test_size += output.getCols();
	//result.allocateHost()->copyToHost()->print("true|false");
}

float mtk::Aggregation::accuracyCalcAccuracy() const{
	if(all_test_size)
		return (float)correct/all_test_size;
	else
		return 0;
}


void mtk::Aggregation::matrixCompareWithTeacher(mtk::MatrixXf &result_matrix, const mtk::MatrixXf &output, const mtk::MatrixXf &teacher){
	deviceMatrixCompareWithTeacher<<<BLOCKS,std::min(THREADS,threads_ceildiv(output.getCols(),BLOCKS))>>>(result_matrix.getDevicePointer(),output.getDevicePointer(),teacher.getDevicePointer(),class_size,batch_size,threads_ceildiv(threads_ceildiv(output.getCols(),BLOCKS),THREADS));
}

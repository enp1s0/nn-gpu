#pragma once

#include "matrix_array.h"
#include "cublas_common.h"
#include "cuda_common.h"
namespace mtk_hidden{
	template<class T>
		__global__ void deviceMap(float* device_ptr_dst,float* device_ptr_src,int max_t){
			int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if(max_t <= tid)
				return;
			device_ptr_dst[tid] = T()(device_ptr_src[tid]);
		}
	template<class T>
		__global__ void deviceMap(float* device_ptr_dst,float* device_ptr_src,float a,int max_t){
			int tid = threadIdx.x + blockIdx.x * blockDim.x;
			if(max_t <= tid)
				return;
			device_ptr_dst[tid] = T(a)(device_ptr_src[tid]);
		}
}
namespace mtk {
	class MatrixFunction{
	public:
		static void copy(cublasHandle_t cublas,mtk::MatrixXf& dst,const mtk::MatrixXf& src);
		static void elementwiseProduct(cublasHandle_t cublas,mtk::MatrixXf& dst,const mtk::MatrixXf& src0,const mtk::MatrixXf& src1);
		template<class T>
			static void map(mtk::MatrixXf& output,const mtk::MatrixXf& input){
				const int BLOCKS = 1 << 7;
				mtk_hidden::deviceMap<T><<<BLOCKS,threads_ceildiv(input.getSize(),BLOCKS)>>>(input.getDevicePointer(),input.getDevicePointer(),input.getSize());
			}
		template<class T>
			static void map(mtk::MatrixXf& output,const mtk::MatrixXf& input,float a){
				const int BLOCKS = 1 << 7;
				mtk_hidden::deviceMap<T><<<BLOCKS,threads_ceildiv(input.getSize(),BLOCKS)>>>(input.getDevicePointer(),input.getDevicePointer(),a,input.getSize());
			}
	};
}

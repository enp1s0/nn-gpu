#include <iostream>
#include "cuda_event.h"
#include "cublas_common.h"
#include "cuda_common.h"
#include "matrix_array.h"
#include "matrix_function.h"

const int MATRIX_SIZE_START = 4;
const int MATRIX_SIZE_END = 1 << 16;
const int CALC = 100;

const int BLOCK_START = 1 << 5;
const int BLOCK_END = 1 << 9;

const int THREAD_START = 1;
const int THREAD_END = 1 << 6;

const float FLOAT_MAX = 100000.0f;

template<class T>
__global__ void deviceMap(float* device_ptr_dst,float* device_ptr_src,int max_t,int loop){
	for(int i = 0;i < loop;i++){
		int tid = (threadIdx.x + blockIdx.x * blockDim.x) * loop + i;
		if(max_t <= tid)
			return;
		device_ptr_dst[tid] = T()(device_ptr_src[tid]);
	}
}
class SFunction{
public:
	__device__ float operator()(float a){
		return __expf(a);
	}
};

int main(){
	std::cout<<"mtk::map<T>() test"<<std::endl;

	mtk::MatrixXf matA,matB;
	mtk::CudaEvent event;
	event.createEvent("start")->createEvent("end");
	for(int i = MATRIX_SIZE_START;i <= MATRIX_SIZE_END;i++){
		matA.setSize(1,i)->allocateDevice()->initDeviceRandom(-1.0f,1.0f);
		matB.setSize(1,i)->allocateDevice()->initDeviceConstant(0.0f);

		int fastest_block = 0;
		int fastest_thread = 0;
		float fastest_time = FLOAT_MAX;

		for(int block = BLOCK_START;block <= BLOCK_END;block<<=1){
			for(int thread = THREAD_START;thread <= THREAD_END;thread<<=1){
				event.recordEvent("start");
				for(int c = 0;c < CALC;c++){
					deviceMap<SFunction><<<block,std::min(thread,threads_ceildiv(matA.getSize(),block))>>>(matB.getDevicePointer(),matA.getDevicePointer(),matA.getSize(),threads_ceildiv(threads_ceildiv(matA.getSize(),block),thread));
				}
				CUDA_HANDLE_ERROR( cudaDeviceSynchronize() );
				event.recordEvent("end");
				float elapsed_time = event.elapsedTime("start","end")/CALC;
				if(fastest_time > elapsed_time){
					fastest_time = elapsed_time;
					fastest_block = block;
					fastest_thread = thread;
				}
			}
		}
		std::cout<<i<<","<<fastest_block<<","<<fastest_thread<<","<<fastest_time<<std::endl;
	}
}

#include "hiddenlayer.h"

using namespace mtk;

const int BLOCKS = 1 << 7;

template<class T>
__global__ void devicaMap(float *device_ptr_dst,float* device_ptr_src,int max_t){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(max_t <= tid)
		return;
	device_ptr_dst[tid] = T()(device_ptr_src[tid]);
}

class ActId{
public:
	__device__ float operator()(float a) const{
		return a;
	}
};


HiddenLayer::HiddenLayer(int input_size,int output_size,int batch_size,std::string layer_name,cublasHandle_t *cublas):
	BaseLayer(input_size,output_size,batch_size,layer_name,cublas)
{}

HiddenLayer::~HiddenLayer(){}

void HiddenLayer::learningBackPropagation(mtk::MatrixXf &next_error, const mtk::MatrixXf &d2, const mtk::MatrixXf *w2){

}

void HiddenLayer::activation(mtk::MatrixXf &output, const mtk::MatrixXf &input) const {
	int matrix_size = input.getCols() * input.getRows();
	devicaMap<ActId><<<BLOCKS,(matrix_size+BLOCKS-1)/BLOCKS>>>(output.getDevicePointer(),input.getDevicePointer(),matrix_size);
}

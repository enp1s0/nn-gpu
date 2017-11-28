#include "cuda_event.h"
#include "cuda_common.h"

using namespace mtk;

CudaEvent* CudaEvent::createEvent(std::string event_name){
	cudaEvent_t event;
	CUDA_HANDLE_ERROR( cudaEventCreate( &event ) );
	events_map.insert(std::make_pair(event_name,event));
	return this;
}

float CudaEvent::elapsedTime(std::string start_event,std::string stop_event){
	float elapsed_time;
	CUDA_HANDLE_ERROR( cudaEventElapsedTime( &elapsed_time, events_map[start_event], events_map[stop_event] ) );
	return elapsed_time;
}

void CudaEvent::recordEvent(std::string event_name){
	CUDA_HANDLE_ERROR( cudaEventRecord( events_map[event_name], 0) );
	CUDA_HANDLE_ERROR( cudaEventSynchronize( events_map[event_name] ) );
}

#pragma once
#include <string>
#include <map>


namespace mtk{
	class CudaEvent{
		std::map<std::string,cudaEvent_t> events_map;
	public:
		CudaEvent* createEvent(std::string event_name);
		float elapsedTime(std::string start_event,std::string stop_event);
		void recordEvent(std::string event_name);
	};
}

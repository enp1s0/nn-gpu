
namespace mtk{
	class ActId{
	public:
		__device__ float operator()(float a) const{
			return a;
		}
	};
	class dActId{
	public:
		__device__ float operator()(float a) const{
			return 1;
		}
	};
	class ActReLU{
	public:
		__device__ float operator()(float a) const{
			return (a>0)?a:0;
		}
	};
	class dActReLU{
	public:
		__device__ float operator()(float a) const{
			return (a>0)?1:0;
		}
	};
}


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
			return (a>0.0f)?a:0.0f;
		}
	};
	class dActReLU{
	public:
		__device__ float operator()(float a) const{
			return (a>0.0f)?1.0f:0.0f;
		}
	};
}

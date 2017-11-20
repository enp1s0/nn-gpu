NVCC=nvcc
NVCCFLAGS= -arch=sm_60 -std=c++11 -lcublas -lcurand
CXX=g++
CXXFLAGS=-std=c++11
OBJDIR=obj
OBJLIST=cuda_common.o main.o cublas_common.o matrix_array.o basenetwork.o hiddennetwork.o softmaxnetwork.o matrix_function.o mnist.o cuda_event.o aggregation.o neuralnetwork.o
OBJS= $(addprefix $(OBJDIR)/, $(OBJLIST))
BIN=exec

$(BIN): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

.SUFFIXES: .o .cu .cpp

.cu.cpp:
	$(NVCC) $(NVCCFLAGS) --cuda $< -o $@

$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@


clean:
	rm -rf $(OBJS)
	rm -rf $(BIN)

cleanlog:
	rm -rf log*

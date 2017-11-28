NVCC=nvcc
NVCCFLAGS= -arch=sm_60 -std=c++11 -lcublas -lcurand
CXX=g++
CXXFLAGS=-std=c++11
SRCDIR=src
OBJDIR=obj
OBJLIST=cuda_common.o main.o cublas_common.o matrix_array.o basenetwork.o hiddennetwork.o softmaxnetwork.o matrix_function.o mnist.o cuda_event.o aggregation.o neuralnetwork.o
OBJS= $(addprefix $(OBJDIR)/, $(OBJLIST))
BIN=nn-gpu

$(BIN): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

$(SRCDIR)/%.cpp: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) --cuda $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@


clean:
	rm -rf $(OBJS)
	rm -rf $(BIN)

cleanlog:
	rm -rf log*

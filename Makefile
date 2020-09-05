BIN=./bin
OBJ=./obj

CCFLAGS= -std=c++11 -O3 -g
NVFLAGS= -ccbin g++-4.8 -arch=sm_30 -Xcompiler -fpic -lineinfo

LD_OPTS= -L/usr/local/cuda/lib64 -lcuda -lcudart

#CCFLAGS += $(NVFLAGS)

all: $(BIN)/benchmark_mies


$(OBJ)/%.o: %.cpp %.h
	$(CXX) -c -o $@ $< $(CCFLAGS)


$(OBJ)/%.o: %.cu %.h
	nvcc -c -o $@ $< $(CCFLAGS) $(NVFLAGS)


$(OBJ)/benchmark_mies.o: benchmark_mies.cpp  benchmark_utils.h
	$(CXX) -c -o $@ $< $(CCFLAGS)


$(BIN)/benchmark_mies: $(OBJ)/benchmark_mies.o  $(OBJ)/sparse.o $(OBJ)/h_mies.o $(OBJ)/d_mies.o
	$(CXX) -o $@ $^ $(CCFLAGS) $(LD_OPTS)


$(OBJ):
	mkdir $@

$(BIN):
	mkdir $@

clean:
	rm $(OBJ)/*.o
	rm $(BIN)/*


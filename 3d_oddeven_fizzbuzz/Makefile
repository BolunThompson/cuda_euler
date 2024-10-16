COMPILER_OPTIONS= -O2 -std=c++20 -Wall -Wextra -Werror
CUDAFLAGS= -arch sm_86
LIBS= 
PROGRAM= fb

.PHONY: all
all: dice

dice: dice.cu
	nvcc $(CUDAFLAGS) -o $(PROGRAM) dice.cu $(LIBS)

.PHONY: clean
	$(RM) $(PROGRAM) gen_data

gen_data: gen_data.cpp
	$(CXX) -o gen_data -O2 -std=c++20 -Wall -Wextra -Wpedantic gen_data.cpp 

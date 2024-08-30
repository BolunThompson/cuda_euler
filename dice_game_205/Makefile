COMPILER_OPTIONS= -O3 -std=c++20 -Wall -Wextra -Werror
CUDAFLAGS=# –Xptxas –dlcm=cg
LIBS= -lcurand
PROGRAM= dice
all:
	nvcc $(CUDAFLAGS) -o $(PROGRAM) dice.cu $(LIBS)

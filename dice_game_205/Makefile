CUDAFLAGS= -O0  -Werror -Wall -Wextra -std=c++20
LIBS= -lcurand
PROGRAM= dice
all:
	nvcc -o $(PROGRAM) dice.cu $(LIBS) $(CUDAFLAGS)

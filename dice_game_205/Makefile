COMPILER_OPTIONS= -O3 -std=c++20 -Wall -Wextra -Werror
CUDAFLAGS= -arch sm_86# –Xptxas –dlcm=cg
LIBS= -lcurand
PROGRAM= dice

.PHONY: all
all: dice

dice: dice.cu
	nvcc $(CUDAFLAGS) -o $(PROGRAM) dice.cu $(LIBS)

.PHONY: clean
	$(RM) $(PROGRAM)

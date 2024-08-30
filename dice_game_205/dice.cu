#include <cstdio>
#include <ctime>
#include <limits>
#include <algorithm>

#include <curand_kernel.h>

#define checkCudaErrors()                                                      \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void game(unsigned long const per_thread, unsigned long seed,
                     unsigned long long *const d_out) {
  const int bt_ind = blockDim.x * blockIdx.x + threadIdx.x;
  if (bt_ind == 0)
    *d_out = 0;

  curandState state;
  curand_init(seed, bt_ind, 0, &state);

  unsigned long outcome = 0;
  for (int i = 0; i < per_thread; ++i) {
    unsigned long peter_res = 0;
    unsigned long colin_res = 0;
    // I'm pretty sure, if this inefficient b/c div ops, it will be optimized
    // TODO: Would manual loop unrolling improve perf? Wouldn't it be automatically unrolled?
    for (int i = 0; i < 9; ++i)
      peter_res += curand(&state) % 4 + 1;
    for (int i = 0; i < 6; ++i)
      colin_res += curand(&state) % 6 + 1;

    outcome += (peter_res > colin_res);
  }

  // I don't think using shared memory would boost performance because
  // while blocking, the thread can switch to other threads to compute.
  // Sure, there might be some idle period at the end as everything is loaded,
  // but that is trivial, I think.
  atomicAdd(d_out, outcome);
}

int main(int argc, char **argv) {
  constexpr unsigned int blocks = 64;
  constexpr unsigned int threads = 1024;
  constexpr unsigned long iters = 1e13;

  // won't exactly lead to iters iterations, but close enough.
  constexpr unsigned long per_thread = iters / (blocks * threads);

  unsigned long long *d_out;
  cudaMalloc(&d_out, sizeof(unsigned long long));
  checkCudaErrors();

  std::printf("game run!\n");
  
  game<<<blocks, threads>>>(per_thread, std::time(NULL), d_out);
  cudaDeviceSynchronize();
  checkCudaErrors();

  auto h_out = new unsigned long;
  cudaMemcpy(h_out, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  // TODO: try with float128
  double result = static_cast<__float128>(*h_out) / iters;
  
  std::printf("ANSWER: %.8f\n", result);
  return 0;
}

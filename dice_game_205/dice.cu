#include <algorithm>
#include <cstdio>
#include <ctime>
#include <limits>

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

__global__ void game(unsigned int const per_thread, unsigned int seed,
                     unsigned long long *const d_out) {
  const int bt_ind = blockDim.x * blockIdx.x + threadIdx.x;
  if (bt_ind == 0)
    *d_out = 0;

  curandState state;
  curand_init(seed, bt_ind, 0, &state);

  unsigned int outcome = 0;
  for (int i = 0; i < per_thread; ++i) {
    // TODO: Would manual loop unrolling improve perf? Wouldn't it be
    // automatically unrolled?

    unsigned int rand = curand(&state);
    unsigned int temp1;
    unsigned int temp2;
    for (int i = 0; i < 4; ++i) {
      temp1 = (temp1 << 8) | ((rand & 0b11) + 1);
      rand >>= 2;
    }
    for (int i = 0; i < 4; ++i) {
      temp2 = (temp2 << 8) | ((rand & 0b11) + 1);
      rand >>= 2;
    }
    auto peter_res = __dp4a(__vadd4(temp1, temp2), 0x01010101U, (curand(&state) & 0b11) + 1);

    for (int i = 0; i < 4; ++i) {
      // TODO: Elim iterated +1
      temp1 = (temp1 << 8) | (curand(&state) % 6 + 1);
    }
    temp2 = 0; // because not all packed values will be replaced
    for (int i = 0; i < 2; ++i) {
      temp2 = (temp2 << 8) | (curand(&state) % 6 + 1);
    }
    auto colin_res = __dp4a(__vadd4(temp1, temp2), 0x01010101U, 0U);

    outcome += (peter_res > colin_res);
  }

  // I don't think using shared memory would boost performance because
  // while blocking, the thread can switch to other threads to compute.
  // Sure, there might be some idle period at the end as everything is loaded,
  // but that is trivial, I think.
  atomicAdd(d_out, outcome);
}

int main(int argc, char **argv) {
  constexpr int blocks = 512;
  constexpr int threads = 512;
  constexpr unsigned long iters = 1e12;

  // won't exactly lead to iters iterations, but close enough.
  constexpr unsigned int per_thread = iters / (blocks * threads);

  unsigned long long *d_out;
  cudaMalloc(&d_out, sizeof(unsigned long long));
  checkCudaErrors();

  std::printf("game run!\n");

  game<<<blocks, threads>>>(per_thread, 0xFAB39, d_out);
  cudaDeviceSynchronize();
  checkCudaErrors();

  auto h_out = new unsigned long long;
  cudaMemcpy(h_out, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  // TODO: would this work with float64?
  double result = static_cast<__float128>(*h_out) / iters;

  std::printf("ANSWER: %.8f\n", result);

  delete h_out;
  cudaFree(d_out);
  checkCudaErrors();
  return EXIT_SUCCESS;
}

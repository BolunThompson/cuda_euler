#include <algorithm>
#include <cstdio>
#include <ctime>
#include <limits>

#include <curand_kernel.h>

// TODO: Fix formatting.
// TODO: Rename this macro
#define checkCudaErrors()                                                      \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

constexpr int BLOCKS = 512;
constexpr int THREADS = 512;
constexpr int SEED = 0xFAB39;
constexpr unsigned long long ITERS = 1e12;

// won't exactly lead to ITERS iterations, but close enough.
constexpr unsigned long long PER_THREAD = ITERS / (BLOCKS * THREADS);
constexpr unsigned long long ACTUAL_ITERS = PER_THREAD * BLOCKS * THREADS;

__device__ __forceinline__ unsigned int
add4sum(unsigned int v1, unsigned int v2, unsigned int initial) {
  return __dp4a(__vadd4(v1, v2), 0x01010101U, initial);
}

__global__ void game(unsigned long long *const d_out) {
  const int bt_ind = blockDim.x * blockIdx.x + threadIdx.x;

  // each thread needs its own state
  // TODO: try differnt rng functions
  curandStatePhilox4_32_10_t state;
  curand_init(SEED, bt_ind, 0, &state);

  unsigned int outcome = 0;
  // First loop, so I generate random colin numbers from 1-4
  #pragma unroll
  for (int i = 0; i < PER_THREAD; ++i) {
    unsigned int rand = curand(&state);
    // TODO: Would using a tensor core speed up this operation?
    // extracts bits 0 and 1 from each byte
    auto temp1 = rand & 0x03030303;
    // extract bits 2 and 3 from each byte
    auto temp2 = (rand & 0x0c0c0c0c) >> 2;
    // the initial value is bit 6 and 7 from the third byte
    auto peter_res = add4sum(temp1, temp2, ((rand & 0xc00000) >> 22) + 9);
    // At this point, I've used 18 bits of the 32 bits of randomness.
    // could I speed this up with << 2 and mod? With int16 ops?
    unsigned long colin_res = (curand(&state) % 6) + (curand(&state) % 6) +
                              (curand(&state) % 6) + (curand(&state) % 6) +
                              (curand(&state) % 6) + (curand(&state) % 6) + 6;

    outcome += (peter_res > colin_res);
  }
  // I don't think using shared memory would boost
  // performance because while blocking, the thread can
  // switch to other threads to compute. Sure, there might
  // be some idle period at the end as everything is
  // loaded, but that is trivial, I think.
  atomicAdd(d_out, outcome);
}

int main(int argc, char **argv) {
  // TODO: Switch blocks/threads count to optimal. See CUDA
  // programming guide.

  auto h_out = new unsigned long long;
  unsigned long long *d_out;
  cudaMalloc(&d_out, sizeof(unsigned long long));
  checkCudaErrors();
  cudaMemset(d_out, 0, sizeof(unsigned long long));
  checkCudaErrors();

  std::printf("game run!\n");

  game<<<BLOCKS, THREADS>>>(d_out);
  cudaDeviceSynchronize();
  checkCudaErrors();

  cudaMemcpy(h_out, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  // In principle, if h_out > 2^53 (9e15), a double could start creating
  // smal calculation errors. A long double gives us exact precision up
  // to 2^63 (9e18), which is enough.
  // Not using ITERS directly because PER_THREAD rounded it down
  double result = static_cast<long double>(*h_out) / ACTUAL_ITERS;

  std::printf("ANSWER: %.8f\n", result);

  delete h_out;
  cudaFree(d_out);
  checkCudaErrors();
  return EXIT_SUCCESS;
}

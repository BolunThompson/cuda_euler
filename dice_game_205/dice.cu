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

constexpr int BLOCKS = 1;
constexpr int THREADS = 1;
constexpr int SEED = 0xFAB39;
constexpr unsigned long long ITERS = 1e3;
// won't exactly lead to ITERS iterations, but close enough.
constexpr unsigned long long PER_THREAD = ITERS / (BLOCKS * THREADS * 3);
constexpr unsigned long long ACTUAL_ITERS = PER_THREAD * (BLOCKS * THREADS * 3);

__device__ __forceinline__ unsigned int
add4sum(unsigned int v1, unsigned int v2, unsigned int initial) {
  return __dp4a(__vadd4(v1, v2), 0x01010101U, initial);
}

__device__ __forceinline__ unsigned int peter_game(unsigned int const rand) {
  // TODO: Would using a tensor core speed up this operation?
  // extracts bits 0 and 1 from each byte
  unsigned int temp1 = rand & 0x03030303;
  // extract bits 2 and 3 from each byte
  unsigned int temp2 = (rand & 0x0c0c0c0c) >> 2;
  // the initial value is bit 6 and 7 from the third byte
  return add4sum(temp1, temp2, ((rand & 0xc00000) >> 22) + 9);
}

__global__ void game(unsigned long long *const d_out) {
  const int bt_ind = blockDim.x * blockIdx.x + threadIdx.x;

  // each thread needs its own state
  curandState state;
  curand_init(SEED, bt_ind, 0, &state);

  unsigned int outcome = 0;
  // First loop, so I generate random colin numbers from 1-4
  // #pragma unroll
  for (int i = 0; i < PER_THREAD * 2; ++i) {
    unsigned int rand = curand(&state);
    unsigned int peter_res = peter_game(rand);
    // At this point, I've used 18 bits of the 32 bits of randomness.

    // extract bits 4 and 5 from each byte
    unsigned int temp1 = (rand & 0x30303030) >> 4;
    // extract bits 6 and 7 from the first two bytes
    unsigned int temp2 = (rand & 0x0000c0c0) >> 6;
    // initial value is 6 to account for a die being from 1-6
    unsigned int colin_res = add4sum(temp1, temp2, 6);
    outcome += (peter_res > colin_res);
    // I've used 30 out of 32 bits of randomness
  }
  // Likewise, but I generate random colin numbers from 5-6
  // #pragma unroll
  // TODO: is there a more compact way to do these calculations?
  for (int i = 0; i < PER_THREAD; ++i) {
    unsigned int rand = curand(&state);
    unsigned int peter_res = peter_game(rand);
    // Extract bit 4 from each byte
    unsigned int temp1 = (rand & 0x10101010) >> 4;
    // Extract bit 6 from the first two bytes
    unsigned int temp2 = (rand & 0x4040) >> 6;
    // initival value is 14 because 6 + 2 * 4, since the dice are from 5-6
    unsigned int colin_res = add4sum(temp1, temp2, 14);
    std::printf("temp1: %#010x, temp2: %#010x\n", temp1, temp2);
    std::printf("peter: %u, colin: %u\n\n", peter_res, colin_res);
    outcome += (peter_res > colin_res);
    // I've used 24 bits of randomness
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

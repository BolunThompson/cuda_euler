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

// Experimentally determined to be reasonable.
// TODO: Tune these.
constexpr int BLOCKS = 8192;
constexpr int THREADS = 256;
constexpr int SEED = 0xFAB39;
constexpr unsigned long long ITERS = 1e12;

// won't exactly lead to ITERS iterations, but close enough.
constexpr unsigned long long PER_THREAD = ITERS / (BLOCKS * THREADS);
constexpr unsigned long long ACTUAL_ITERS = PER_THREAD * BLOCKS * THREADS;

__device__ __forceinline__ unsigned int
add4sum(unsigned int v1, unsigned int v2, unsigned int initial) {
  return __dp4a(__vadd4(v1, v2), 0x01010101U, initial);
}

// TODO: Profile. The runpod container doesn't give me access to one,
//       so I need to buy or borrow a computer with an NVIDIA GPU.
//       However, the main issue is certainly the inner loop.
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

    // I have spent a day attempting to speed this up; however, the naive
    // approach stil seems to be the best. The speed changes are from before and
    // after. I sometimes made (hopefully) minor changes in between.
    // ATTEMPTS:
    // 1. Replacing these with int16 and doing half as many generations. The
    //    accuracy becomes worse (last 3 digits vary, not just last two),
    //    so this is unideal. Does speed it up, however (21s ~> 14s),
    // 2. Using add4sum and bitshifts. Slightly slower (28s ~> 29s)
    // 3. Using __vcmpgeu4 and __vminu4 (IIRC) to check whether each byte in
    //    the 32 bit value is a 7 or 8 (I masked the higer bits, so I just
    //    had to convert the value from 0-7 to 0-5), then replacing the too high
    //    ones with one from a new rand. Each additional check took
    //    approximately 4 seconds, and the program took 4 seconds to startup.
    //    However, for suitable accuracy I needed to do 9 checks, meaning that
    //    it would take 40 seconds -- slower than the unvectorized approach. I
    //    didn't try whether I should do a certain number of checks always and
    //    then start checking whether 6s and 7s has been purged from the value.
    //    (for future reference: the funciton (1 - (.25 ** (x*2))) ** 128) where
    //    x is the number of high value checks).

    // 4. Manually checking each part of the byte to see if it's too big, and
    //    replacing the upper two 0b11 bits (of the 3 in the 8 bit generated
    //    value) with two from another curand. This only necessitates 2 rand
    //    calls, and I can avoid branching within the "if too big" branch with a
    //    clever __clz call†. However, all the branching leads to wasted
    //    instructions in all the threads of the warp. When I tried it, it took
    //    16s, but keep in mind things were generally running faster then, so I
    //    don't know if that or the naive approach would be faster. I somehwat
    //    doubt it would have. (TODO: If I come back to this, test again).

    //    † NOT the extra rand_val and count the number of leading zero
    //    bits with __clz, then & that count with 1. This calculates the right
    //    shift needed to get rid of leading 0b11 (too big) in rand_val, so the
    //    leading bits could be used to replace the upper two bits in a given
    //    too big value. I never tested this algorithm -- in all honesty, it
    //    could be slower than just doing a while loop and testing.

    // If I were doing this again, I'd be more careful to keep track of profiler
    // results and specific code iteration. (TODO: Is there a tool for this?)

    unsigned long colin_res = (curand(&state) % 6) + (curand(&state) % 6) +
                              (curand(&state) % 6) + (curand(&state) % 6) +
                              (curand(&state) % 6) + (curand(&state) % 6) + 6;

    outcome += (peter_res > colin_res);
  }
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

  std::printf("ANSWER: %.7f\n", result);

  delete h_out;
  cudaFree(d_out);
  checkCudaErrors();
  return EXIT_SUCCESS;
}

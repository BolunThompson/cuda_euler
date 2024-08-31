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

#define REPLACE_TOO_BIG_RAND(rand_var, test, rep_bits)                         \
  do {                                                                         \
    if ((rand_var & (test)) == (test)) {                                       \
      std::printf(                                                             \
          "inputs:\nrand_var: %0#010x\ntest: %0#010x\nrep_bits: %0#010x\n\n",  \
          rand_var, test, rep_bits);                                           \
      while ((rep_rand & 0x6) == 0x6)                                          \
        rep_rand >>= 3;                                                        \
      if (rep_rand == 0xc)                                                     \
        rep_rand = curand(&state) | 0xc0000000;                                \
      rand_var = (rand_var & ~(rep_bits)) | (rep_rand & 0x7);                  \
      rep_rand >>= 3;                                                          \
      std::printf("outputs:\nrand_var: %0#010x\ntest: %0#010x\nrep_bits: "     \
                  "%0#010x\n\n\n",                                             \
                  rand_var, test, rep_bits);                                   \
    }                                                                          \
  } while (0)

__global__ void game(unsigned int const per_thread, unsigned int const seed,
                     unsigned long long *const d_out) {
  const int bt_ind = blockDim.x * blockIdx.x + threadIdx.x;
  if (bt_ind == 0)
    *d_out = 0;

  // each thread needs its own state
  curandState state;
  curand_init(seed, bt_ind, 0, &state);

  unsigned int outcome = 0;
  for (int i = 0; i < per_thread; ++i) {
    // TODO: Replace repeats with preprocessor macros
    unsigned int rand = curand(&state);
    unsigned int temp1 =
        rand & 0x03030303; // extracts bits 0 and 1 from each byte
    unsigned int temp2 =
        (rand & 0x0c0c0c0c) >> 2; // extract bits 2 and 3 from each byte
    // the initial value is bit 4 and 5 from the first byte
    // TODO: Would using a tensor core speed up this operation?
    auto peter_res =
        __dp4a(__vadd4(temp1, temp2), 0x01010101U, (rand & 0x30) >> 4) + 9;
    // At this point, I've only used 18 bits of the 32 bits of randomness
    // provided If I simply generate another 32 bits and discard values of 6 or
    // 7, I'd on average be able to generate 11 more numbers. I need 6. However,
    // there's an uncomfortably high chance of [] that I don't get all the
    // values I need. In that case, the chance of 6 or more of those 11 numbers
    // being a 6 or 7 is 3.43% (calculation done via the binomial theorem).
    // Therefore, for some threads, another random number will be generated.

    // NOTE: The actual chances above are higher than calculated since I'm
    // throwing away the highest bit of randomness in each byte (and the
    // remaining) two in the first. I don't know if it's faster to extract those
    // or to generate somewhat more random numbers (TODO profile).

    // When one value is above the range (6 or 7), that value is replaced with
    // the next 3 valid bits in rep_rand. When replacements == sentinel, all
    // the values have been exhausted (will happen in 7.81% of runs) a new
    // replacements will be generated.
    rand = curand(&state);
    unsigned int rep_rand = curand(&state) | 0xc0000000;
    // extract the first 3 bits of each byte
    temp1 = rand & 0x07070707;
    // extract the second 3 bits of the last two bytes
    temp2 = (rand & 0x00003838) >> 3;
    // if bits 1 and 2 in the first byte are set (meaning that the value
    // equals 6 or 7), replace bits 0-2 with the next 3 bits in rep_rand.
    REPLACE_TOO_BIG_RAND(temp1, 0x6, 0x7);
    REPLACE_TOO_BIG_RAND(temp1, 0x6 << 8, 0x7 << 8);
    REPLACE_TOO_BIG_RAND(temp1, 0x6 << 16, 0x7 << 16);
    REPLACE_TOO_BIG_RAND(temp1, 0x6 << 24, 0x7 << 24);
    REPLACE_TOO_BIG_RAND(temp2, 0x6, 0x7);
    REPLACE_TOO_BIG_RAND(temp2, 0x6 << 8, 0x7 << 8);
    auto colin_res = __dp4a(__vadd4(temp1, temp2), 0x01010101U, 6U);
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
  constexpr int blocks = 1;
  constexpr int threads = 1;
  constexpr unsigned long iters = 1e3;
  // won't exactly lead to iters iterations, but close
  // enough.
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

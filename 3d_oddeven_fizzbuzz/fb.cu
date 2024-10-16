// Loads the random data as a 3d grid and
// determines where there are an odd or even number of adjascent fizzes and buzzes and
// then writes that encoded in binary to fb.bin
// Somewhat like a convalutional kernel

#include <algorithm>
#include <cstdio>
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

// Experimentally determined to be reasonable.
// TODO: Tune these.
// The native approach of having BLOCKS = ITERS and THREADS = 1 doesn't work
// due to scheduler overhead.
constexpr int BLOCKS = 8192;
constexpr int THREADS = 256;

// TODO: Profile. The runpod container doesn't give me access to one,
//       so I need to buy or borrow a computer with an NVIDIA GPU.
__global__ void fboe3d(unsigned int* __restrict__ const data) {
}

int main(int argc, char **argv) {
  auto h_out = new unsigned long long;
  cudaMalloc(&d_out, sizeof(unsigned long long));
  checkCudaErrors();
  cudaMemset(d_out, 0, sizeof(unsigned long long));
  checkCudaErrors();

  std::printf("game run!\n");

  fboe3d<<<BLOCKS, THREADS>>>(d_out);
  cudaDeviceSynchronize();
  checkCudaErrors();

  cudaMemcpy(h_out, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

  // cleanup:

  delete h_out;
  cudaFree(d_out);
  checkCudaErrors();
  return EXIT_SUCCESS;
}

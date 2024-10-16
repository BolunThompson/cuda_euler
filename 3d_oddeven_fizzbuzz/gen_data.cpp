#include <vector>
#include <random>
#include <climits>
#include <algorithm>
#include <functional>
#include <iostream>
#include <fstream>

using random_bytes_engine = std::independent_bits_engine<
    std::default_random_engine, std::numeric_limits<int>::digits, unsigned int>;

constexpr unsigned int seed = 0xFAB39;

// Takes the number of bytes of integers to generate
// Takes about 2:30 mins on an i7-1280P
// If I want to generate larger datasets, I should use a faster (gpu?)
// sorting algorithm
int main(int argc, char *argv[]) {
    size_t byte_len = argc > 1 ? std::stoull(argv[1]) : 0xFFFFFFFF;
    random_bytes_engine rbe(seed);
    std::vector<int> data(byte_len / sizeof(int));
    std::printf("Generation Start!\n");
    std::generate(begin(data), end(data), std::ref(rbe));
    std::printf("Generation End!\n");

    std::ofstream random_out("random.bin", std::ios::binary);
    random_out << byte_len;
    random_out.write(reinterpret_cast<const char*>(&data[0]), byte_len);
    random_out.close();
    if (!random_out) {
      std::cout << "ERROR: random.bin write error" << std::endl;
      return 1;
    }
    std::cout << "random.bin written" << std::endl;

    // TODO: Try -stdpar=gpu for parallel sorting on cuda with nvcc
    std::sort(data.begin(), data.end());
    for (int i = 0; i < 10; ++i)
      std::cout << data[i] << std::endl;
   
    std::cout << "random_sorted.bin sorted" << std::endl;

    std::ofstream random_sout("random_sorted.bin", std::ios::binary);
    random_sout << byte_len;
    random_sout.write(reinterpret_cast<const char*>(&data[0]), byte_len);
    random_sout.close();

    if (!random_out) {
      std::clog << "ERROR: random.bin written; random_sorted.bin write error" << std::endl;
      return 1;
    }

    std::cout << "random_sorted.bin written" << std::endl;

    return 0;
}

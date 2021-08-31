#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

constexpr unsigned int DIGITS_PER_STEP = 2;
constexpr unsigned int VALUES_PER_DIGIT = (1 << DIGITS_PER_STEP);


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Length: " << n << std::endl;

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    {
        unsigned int workGroupSize = 128;

        std::stringstream ss;

        ss << "-D WORK_SIZE=" << workGroupSize << " -D DIGITS_PER_STEP=" << DIGITS_PER_STEP
            << " -D VALUES_PER_DIGIT=" << VALUES_PER_DIGIT;

        std::string defines = ss.str();

        ocl::Kernel local_sum(radix_kernel, radix_kernel_length, "local_sum", defines);
        local_sum.compile();

        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix", defines);
        radix.compile();

        unsigned int groups_count = (n + workGroupSize - 1) / workGroupSize;
        unsigned int global_work_size = groups_count * workGroupSize;
        unsigned int sums_size = VALUES_PER_DIGIT * (groups_count + 1);

        gpu::gpu_mem_32u sums_gpu;
        sums_gpu.resizeN(sums_size);

        gpu::gpu_mem_32u indexes;
        indexes.resizeN(n);

        gpu::gpu_mem_32u as_buffer;
        as_buffer.resizeN(n);

        constexpr unsigned int initialMask = 0xFFFFFFFF >> (sizeof(unsigned int) * 8 - DIGITS_PER_STEP);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            std::vector<unsigned int> sums(sums_size, 0);
            sums_gpu.writeN(sums.data(), sums_size);

            t.restart();

            unsigned int step = 0;
            for (unsigned int digit = 0; digit < 32; digit += DIGITS_PER_STEP) {
                local_sum.exec(gpu::WorkSize(workGroupSize, global_work_size),
                    as_gpu, indexes, sums_gpu, initialMask, step, n);

                sums_gpu.readN(sums.data(), sums_size);

                
                for (unsigned int i = VALUES_PER_DIGIT; i < sums_size; ++i) {
                    sums[i] += sums[i - VALUES_PER_DIGIT];
                }

                sums_gpu.writeN(sums.data(), sums_size);

                radix.exec(gpu::WorkSize(workGroupSize, global_work_size),
                    as_gpu, as_buffer, indexes, sums_gpu, initialMask, step, n);

                as_gpu.swap(as_buffer);

                step += DIGITS_PER_STEP;
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1e6) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << "GPU: " << (n * sizeof(float) / 1e9) / t.lapAvg() << " GB/s" << std::endl;
        as_gpu.readN(as.data(), n);
    }


    return 0;
}

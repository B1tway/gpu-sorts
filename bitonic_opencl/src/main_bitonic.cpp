﻿#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/bitonic_cl.h"

#include <vector>
#include <iostream>
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


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
	unsigned int n = 32 * 1024 * 1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Length:" << n << std::endl;


    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);

    {
        ocl::Kernel bitonic(bitonic_kernel, bitonic_kernel_length, "bitonic");
		bitonic.compile();
		ocl::Kernel bitonic_local(bitonic_kernel, bitonic_kernel_length, "bitonic_local");
		bitonic_local.compile();

        timer t;
		unsigned int workGroupSize = 256;
		unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

		// Ищем наименьшую степень двойки, большую нашего n
		unsigned int max_size = 2;
		while (max_size < n) max_size <<= 1;

        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();

			// Начинаем с сортировки подмассивов длины 2
			unsigned int size = 2;

			while (size <= max_size) {
				// Размер части массива одного "тона"
				unsigned int batch_size = size >> 1;

				// Обработка больших кусков
				while (2 * batch_size > workGroupSize) {
					bitonic.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, batch_size, size, n);
					batch_size >>= 1;
				}

				// Локальная обработка
				bitonic_local.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, batch_size, size, n);
				size <<= 1;
			}

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1e6) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << "GPU: " << (n * sizeof(float) / 1e9) / t.lapAvg() << " GB/s" << std::endl;
        as_gpu.readN(as.data(), n);
    }

    
    return 0;
}

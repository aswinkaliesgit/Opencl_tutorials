#define __CL_ENABLE_EXCEPTIONS
#define CL_TARGET_OPENCL_VERSION 220

#include "cl.hpp"
#include "util.hpp" 
#include "err_code.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>
#include <chrono>
#define LENGTH 2048
#define tolerance 1
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

bool verify_results(std::vector<float>*, std::vector<float>*, std::vector<float>*, int);

int main(void) {
    // Initialize vectors and buffers
    std::vector<float> h_a(LENGTH * LENGTH);
    std::vector<float> h_b(LENGTH * LENGTH);
    std::vector<float> h_c(LENGTH * LENGTH, 0.0f);
    cl::Buffer d_a, d_b, d_c;

    // Initialize OpenCL context, command queue, and program
    cl::Context context(DEVICE);
    cl::CommandQueue queue(context);
    cl::Program program(context, util::loadProgram("tiled_matrix.cl"), true);

    // Generate random values for matrices A and B
    for (int i = 0; i < LENGTH * LENGTH; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Normal matrix multiplication
    auto start_normal = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < LENGTH; i++) {
        for (int j = 0; j < LENGTH; j++) {
            float sum = 0.0f;
            for (int k = 0; k < LENGTH; k++) {
                sum += h_a[i * LENGTH + k] * h_b[k * LENGTH + j];
            }
            h_c[i * LENGTH + j] = sum;
        }
    }
    auto end_normal = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_normal = end_normal - start_normal;

    // Tiled matrix multiplication
    auto start_tiled = std::chrono::high_resolution_clock::now();
    d_a = cl::Buffer(context, begin(h_a), end(h_a), true);
    d_b = cl::Buffer(context, begin(h_b), end(h_b), true);
    d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH * LENGTH);
    auto tiled_matrix = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "tiled_matrix");
    tiled_matrix(cl::EnqueueArgs(queue, cl::NDRange(LENGTH, LENGTH), cl::NDRange(16, 16)), d_a, d_b, d_c, LENGTH);
    queue.finish();
    cl::copy(queue, d_c, begin(h_c), end(h_c));
    auto end_tiled = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_tiled = end_tiled - start_tiled;

    // Verify results
    if (verify_results(&h_a, &h_b, &h_c, LENGTH)) {
        std::cout << "Matrix multiplication is correct." << std::endl;
    } else {
        std::cout << "Matrix multiplication is incorrect." << std::endl;
    }

    // Print timing results
    std::cout << "Normal matrix multiplication time: " << elapsed_normal.count() << " seconds" << std::endl;
    std::cout << "Tiled matrix multiplication time: " << elapsed_tiled.count() << " seconds" << std::endl;

    return 0;
}

bool verify_results(std::vector<float>* a, std::vector<float>* b, std::vector<float>* c, int N) {
    int correct = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float tmp = 0.0f;
            for (int k = 0; k < N; k++) {
                tmp += (*a)[i * N + k] * (*b)[k * N + j];
            }
            float check = (*c)[i * N + j];
            if (std::abs(tmp - check) < tolerance) {
                correct++;
            }
        }
    }

    return (correct == N * N);
}

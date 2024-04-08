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
#define LENGTH 1024
#define tolerance 1
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

bool verify_results(std::vector<float>*, std::vector<float>*, std::vector<float>*, int);

int main(void)
{
  std::vector<float> h_a(LENGTH*LENGTH);
  std::vector<float> h_b(LENGTH*LENGTH);
  std::vector<float> h_c(LENGTH*LENGTH,0.0f);

  cl::Buffer d_a;
  cl::Buffer d_b;
  cl::Buffer d_c;

  cl::Context context(DEVICE);
  cl::CommandQueue queue(context);
  cl::Program program(context, util::loadProgram("naive_mul.cl"), true);
  

  for(int i=0;i<LENGTH*LENGTH;i++)
  {
    h_a[i]= rand()/(float) RAND_MAX;
    h_b[i]= rand()/(float)RAND_MAX;
  }

 auto naive_mul = cl::make_kernel<cl::Buffer,cl::Buffer,cl::Buffer,int,cl::LocalSpaceArg>(program,"naive_mul");

 d_a   = cl::Buffer(context, begin(h_a), end(h_a), true);
 d_b   = cl::Buffer(context, begin(h_b), end(h_b), true);

 d_c  = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) *LENGTH*LENGTH);
cl::LocalSpaceArg localmem = cl::Local(sizeof(float) * LENGTH);
          naive_mul(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(LENGTH),cl::NDRange(LENGTH/64)), 
            d_a,
            d_b,
            d_c,
            LENGTH,localmem);
            queue.finish();
cl::copy(queue, d_c, begin(h_c), end(h_c));

if (verify_results(&h_a, &h_b, &h_c, LENGTH)) {
    printf("Matrix multiplication is correct.\n");
} else {
    printf("Matrix multiplication is incorrect.\n");
}



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
            //printf("%f %f\n",tmp,check);
 
            if (std::abs(tmp - check) < tolerance) {
                correct++;
            }
        }
    }

    return (correct == N * N);
}


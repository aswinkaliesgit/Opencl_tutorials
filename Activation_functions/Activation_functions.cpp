#include <iostream>
#include <vector>
#include <cl.hpp>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <iomanip> 

#define SIZE 1024

int main(void) {
    std::vector<float> input_tensor(SIZE);
    std::vector<float> output_tensor(SIZE);

    for (int i = 0; i < input_tensor.size(); i++) {
        input_tensor[i] = i + 1;
    }

    int function = 4;

    switch (function) {
        case 1:
            // sigmoid
            for (int i = 0; i < input_tensor.size(); i++) {
                float temp = 1 + exp(-1 * input_tensor[i]);
                temp = 1 / temp;
                output_tensor[i] = temp;
            }
            break;

        case 2:
            // relu
            for (int i = 0; i < input_tensor.size(); i++) {
                output_tensor[i] = std::max(0.0f, input_tensor[i]);
            }
            break;

        case 3:
            // tanh
            for (int i = 0; i < input_tensor.size(); i++) {
                float temp = 1 + exp(-2 * input_tensor[i]);
                temp = 2 / temp - 1;
                output_tensor[i] = temp;
            }
            break;

        case 4:
            // relu 6
            for (int i = 0; i < input_tensor.size(); i++) {
                output_tensor[i] = std::min(std::max(input_tensor[i], 0.0f), 6.0f);
            }
            break;

        case 5:
            // GELU
            for (int i = 0; i < input_tensor.size(); i++) {
                float temp = sqrt(2 / M_PI);
                temp = temp * (input_tensor[i] + 0.044715 * input_tensor[i] * input_tensor[i] * input_tensor[i]);
                temp = 1 + exp(-2 * temp);
                temp = 2 / temp - 1;
                temp = 0.5 * input_tensor[i] * (1 + temp);
                output_tensor[i] = temp;
            }
            break;

        case 6:
            // SiLU
            for (int i = 0; i < input_tensor.size(); i++) {
                float temp = 1 + exp(-1 * input_tensor[i]);
                temp = 1 / temp;
                output_tensor[i] = temp * input_tensor[i];
            }
            break;
    }

    // opencl part
    std::string sourceCode;
    std::ifstream sourceFile("Activation_functions.cl");

    if (!sourceFile.is_open()) {
        std::cerr << "Failed to open OpenCL source file." << std::endl;
        return 1;
    }

    sourceCode.assign(std::istreambuf_iterator<char>(sourceFile),
                      (std::istreambuf_iterator<char>()));

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    cl::Platform platform = platforms[0];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    cl::Context context(devices);
    cl::CommandQueue queue(context, devices[0]);
    std::vector<float> final_out(SIZE);
    cl::Program program(context, sourceCode);
    program.build(devices);
    cl::Buffer cl_flat_vector(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input_tensor.size(), input_tensor.data());
    cl::Buffer cl_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * (input_tensor.size()));
    cl::Kernel kernel(program, "Activation_functions");
    kernel.setArg(0, cl_flat_vector);
    kernel.setArg(1, cl_output);
    kernel.setArg(2, function);
    cl::NDRange global_size(SIZE);
    cl_int err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, cl::NullRange);

    if (err != CL_SUCCESS) {
        std::cerr << "Error enqueuing kernel. Error code: " << err << std::endl;
        return {};
    }

    queue.enqueueReadBuffer(cl_output, CL_TRUE, 0, sizeof(float) * final_out.size(), final_out.data());
    std::cout << std::fixed << std::setprecision(4);

    for (int i = 0; i < input_tensor.size(); i++) {
        std::cout << final_out[i] << " ";
    }

    return 0;
}

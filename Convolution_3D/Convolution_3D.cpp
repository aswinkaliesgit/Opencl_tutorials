#include <iostream>
#include <vector>
#include <cl.hpp>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#define INPUT_SIZE 10
#define DEPTH 5
int main(void) { 
    std::vector<std::vector<std::vector<float>>> input_tensor(DEPTH, std::vector<std::vector<float>>(INPUT_SIZE, std::vector<float>(INPUT_SIZE)));
    std::vector<std::vector<float>> kernel_filter = {{2, 2, 2}, {2, 2, 2}, {2, 2, 2}};
    std::vector<float> input_tensor_flattened((INPUT_SIZE)*(INPUT_SIZE)*DEPTH);
    std::vector<float> kernel_filter_flattened = {2, 2, 2, 2, 2, 2, 2, 2, 2};

    for (int i = 0; i < DEPTH; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            for (int k = 0; k < INPUT_SIZE; ++k) {
                input_tensor[i][j][k] = 1;
                input_tensor_flattened[i*INPUT_SIZE*INPUT_SIZE + j*INPUT_SIZE+k]=1;
            }
        }
    }

    int kernel_size = kernel_filter.size();
    int stride = 1;

    int input_height = input_tensor[0].size();
    int input_width = input_tensor[0][0].size();
    int input_depth = DEPTH;
    int output_height = input_height;
    int output_width = input_width;
    int output_depth = DEPTH;
    std::vector<std::vector<std::vector<float>>> output_tensor(DEPTH, std::vector<std::vector<float>>(INPUT_SIZE, std::vector<float>(INPUT_SIZE)));
    std::vector<float> output_tensor_flattened((INPUT_SIZE) * (INPUT_SIZE)*DEPTH);

    for (int c = 0; c < DEPTH; ++c) {
        for (int i = 0; i < input_height; ++i) {
            for (int j = 0; j < input_width; ++j) {
                float sum=0.0f;
                for (int m = 0; m < kernel_size; ++m) {
                    for (int n = 0; n < kernel_size; ++n) {
                        int ii = i + m - kernel_size / 2;
                        int jj = j + n - kernel_size / 2;
                        if (ii >= 0 && ii < INPUT_SIZE && jj >= 0 && jj < INPUT_SIZE) {
                            sum += kernel_filter_flattened[m * kernel_size + n] * input_tensor_flattened[(ii * INPUT_SIZE + jj) * DEPTH + c];
                        }
                    }
                }
                output_tensor_flattened[c*INPUT_SIZE*INPUT_SIZE+i*INPUT_SIZE+j]=sum;
                
            }
        }
    }
//open cl part 

std::string sourceCode;
    std::ifstream sourceFile("Convolution_3D.cl");
    if (!sourceFile.is_open()) {
        std::cerr << "Failed to open OpenCL source file." << std::endl;
        return 1;
    }
    sourceCode.assign(
        std::istreambuf_iterator<char>(sourceFile),
        (std::istreambuf_iterator<char>()));

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    cl::Platform platform = platforms[0];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    cl::Context context(devices);
    cl::CommandQueue queue(context, devices[0]);
    cl::Program program(context, sourceCode);
    program.build(devices);
    cl::Buffer cl_input_tensor_flattened(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) *input_tensor_flattened.size() ,input_tensor_flattened.data());
    cl::Buffer cl_output_tensor_flattened(context, CL_MEM_WRITE_ONLY, sizeof(float) *(output_height*output_width*DEPTH));
    cl::Buffer cl_kernel_flattened(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(float)*kernel_filter_flattened.size(),kernel_filter_flattened.data());
    cl::Kernel kernel(program, "Convolution_3D");
    kernel.setArg(0, cl_input_tensor_flattened);
    kernel.setArg(1, cl_output_tensor_flattened);
    kernel.setArg(2,input_height);
    kernel.setArg(3,input_width);
    kernel.setArg(4,output_height);
    kernel.setArg(5,output_width);
    kernel.setArg(6,kernel_size);
    kernel.setArg(7,cl_kernel_flattened);
    cl::NDRange global_size(DEPTH,input_height,input_width);
    cl_int err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, cl::NullRange);
    if (err != CL_SUCCESS) {
        std::cerr << "Error enqueuing kernel. Error code: " << err << std::endl;
        return {};
    }
    std::vector<float> final_out(output_height*output_width*DEPTH);
   queue.enqueueReadBuffer(cl_output_tensor_flattened, CL_TRUE, 0, sizeof(float) * final_out.size(),final_out.data());
         for (int i = 0; i < DEPTH; ++i) {
            std::cout<<"depth "<<i<<"\n";
        for (int j = 0; j < INPUT_SIZE; ++j) {
            for (int k = 0; k < INPUT_SIZE; ++k) {
        
             std::cout<<final_out[i*INPUT_SIZE*INPUT_SIZE + j*INPUT_SIZE+k]<<" ";
            }
            std::cout<<"\n";
        }
        std::cout<<"\n";
    }
    

}
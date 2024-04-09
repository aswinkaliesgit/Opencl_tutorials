#include <iostream>
#include <vector>
#include <cl.hpp>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#define INPUT_SIZE 4
int main(void)
{

   std::vector<std::vector<float>> input_tensor(INPUT_SIZE , std::vector<float>(INPUT_SIZE));
    std::vector<std::vector<float>> kernel_filter = {{2, 2, 2}, {2, 2, 2}, {2, 2, 2}};
    std::vector<float> input_tensor_flattened((INPUT_SIZE) * (INPUT_SIZE));
    std::vector<float> kernel_filter_flattened = {2, 2, 2, 2, 2, 2, 2, 2, 2};

    for (int i = 0; i < INPUT_SIZE ; i++) {
        for (int j = 0; j < INPUT_SIZE ; j++) {
            input_tensor[i][j] = 1;
            input_tensor_flattened[i*INPUT_SIZE+ j] = 1;
        }
    }

    int kernel_size = kernel_filter.size();
    int stride = 1;

    int input_height = input_tensor.size();
    int input_width = input_tensor[0].size();
    int output_height = input_height;
    int output_width = input_width;


    std::vector<std::vector<float>> output_tensor(output_height, std::vector<float>(output_width, 0));
    std::vector<float> output_tensor_flattened(output_height * output_width);

    for (int i = 0; i < output_height; ++i) {
        int start_point = i - (kernel_size / 2);
        for (int j = 0; j < output_width; ++j) {
            int end_point = j - (kernel_size / 2);
            float sum = 0;
            for (int k = 0; k < kernel_size; ++k) {
                for (int l = 0; l < kernel_size; ++l) {
                    if ((start_point + k >= 0 && start_point + k < INPUT_SIZE) && (end_point + l >= 0 && end_point + l < INPUT_SIZE)) {
                        sum += input_tensor_flattened[(start_point + k)*INPUT_SIZE+ end_point+l] * kernel_filter_flattened[k*kernel_size+l];
                    }
                }
            }
            output_tensor_flattened[i*output_width+j] = sum;
        }
    }

  //openclpart
   std::string sourceCode;
    std::ifstream sourceFile("Convolution_2D.cl");
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
    cl::Buffer cl_output_tensor_flattened(context, CL_MEM_WRITE_ONLY, sizeof(float) *(output_height*output_width));
    cl::Buffer cl_kernel_flattened(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,sizeof(float)*kernel_filter_flattened.size(),kernel_filter_flattened.data());
    cl::Kernel kernel(program, "Convolution_2D");
    kernel.setArg(0, cl_input_tensor_flattened);
    kernel.setArg(1, cl_output_tensor_flattened);
    kernel.setArg(2,input_height);
    kernel.setArg(3,input_width);
    kernel.setArg(4,output_height);
    kernel.setArg(5,output_width);
    kernel.setArg(6,kernel_size);
    kernel.setArg(7,cl_kernel_flattened);
    cl::NDRange global_size(input_height,input_width);
    cl_int err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, cl::NullRange);
    if (err != CL_SUCCESS) {
        std::cerr << "Error enqueuing kernel. Error code: " << err << std::endl;
        return {};
    }
    std::vector<float> final_out(output_height*output_width);
   queue.enqueueReadBuffer(cl_output_tensor_flattened, CL_TRUE, 0, sizeof(float) * final_out.size(),final_out.data());
   int count=0;
    for(int i=0;i<output_height;i++)
    {
        for(int j=0;j<output_width;j++)
        {
        if(final_out[i*output_width+j]==output_tensor_flattened[i*output_width+j])
             count++;
        }
    }

if(count==output_height*output_width) std::cout<<"correct";

    // for (int i = 0; i < output_height; ++i) {
    //     for (int j = 0; j < output_width; ++j) {
    //         std::cout << final[i*output_width+j] << "\t";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}
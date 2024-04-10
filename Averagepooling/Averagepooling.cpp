#include <iostream>
#include <vector>
#include <cl.hpp>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
int main(void)
{
    std::vector<std::vector<float>> input_tensor(1024,std::vector<float>(1024));
    for(int i=0;i<1024;i++)
    {
        for(int j=0;j<1024;j++)
        {
            input_tensor[i][j]=rand()/100.0;
        }
    }

    int kernel_size=8;
    int stride=2;

    int input_height=input_tensor.size();
    int input_width = input_tensor[0].size();
    int output_height=(input_height-kernel_size)/stride+1;
    int output_width= (input_width-kernel_size)/stride+1;
    int padding_height= kernel_size-(input_height-(output_height-1)*stride);
    int padding_width =  kernel_size-(input_width-(output_width-1)*stride);

    // std::cout<<output_height<<" "<<output_width;

   std::vector<std::vector<float>> output_tensor(output_height, std::vector<float>(output_width, 0.0f));
float local_max;
    for(int i=0;i<output_height;i++)
    {
        for(int j=0;j<output_width;j++)
        {
            local_max=-100;
             for(int k=0;k<kernel_size;k++)
             {
                for(int l=0;l<kernel_size;l++)
                {
                   local_max = std::max(local_max,input_tensor[i*stride+k][j*stride+l]);
                }
             }
             output_tensor[i][j]=local_max;
        }
    }

std::vector<float> input_tensor_flattened(input_height*input_width);
std::vector<float> output_tensor_flattened(output_height*output_width);
for(int i=0;i<input_height;i++)
{
    for(int j=0;j<input_width;j++)
    {
        input_tensor_flattened[i*input_width+j]=input_tensor[i][j];
    }
}

    for (int i = 0; i < output_height; i++)
    {
        for (int j = 0; j < output_width; j++)
        {
            local_max = 0.0f;
            for (int k = 0; k < kernel_size; k++)
            {
                for (int l = 0; l < kernel_size; l++)
                {
                    int input_index = (i * stride + k) * input_width + (j * stride + l);
                    local_max +=input_tensor_flattened[input_index]; 
                }
            }
            int output_index = i * output_width + j;
            output_tensor_flattened[output_index] = local_max/(kernel_size*kernel_size);
        }
    }
 std::string sourceCode;
    std::ifstream sourceFile("Averagepooling.cl");
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
    cl::Kernel kernel(program, "Averagepooling");
    kernel.setArg(0, cl_input_tensor_flattened);
    kernel.setArg(1, cl_output_tensor_flattened);
    kernel.setArg(2,input_height);
    kernel.setArg(3,input_width);
    kernel.setArg(4,output_height);
    kernel.setArg(5,output_width);
    kernel.setArg(6,kernel_size);
    kernel.setArg(7,stride);
    cl::NDRange global_size(output_height,output_width);
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


}
#include <iostream>
#include <vector>
#include <cl.hpp>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#define GROUP_SIZE 8
#define TOTAL_SIZE 8
   const int dim1 =  1;
    const int dim2 = 2;
    const int dim3 = 2;
    const int dim4 = 2;

int main(void){
    int ****array4d = (int ****)malloc(dim1 * sizeof(int ***));
    for (int i = 0; i < dim1; ++i) {
        array4d[i] = (int ***)malloc(dim2 * sizeof(int **));
        for (int j = 0; j < dim2; ++j) {
            array4d[i][j] = (int **)malloc(dim3 * sizeof(int *));
            for (int k = 0; k < dim3; ++k) {
                array4d[i][j][k] = (int *)malloc(dim4 * sizeof(int));
            }
        }
    }

    // Initialize and calculate sum
    int product_test = 1;
    int flag=0;
    std::vector<int> flat_vector(dim1 * dim2 * dim3 * dim4);
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            for (int k = 0; k < dim3; ++k) {
                for (int l = 0; l < dim4; ++l) {
                    array4d[i][j][k][l] =2;
                 int tmp= array4d[i][j][k][l];
                    product_test= tmp*product_test;
                    flat_vector[i * dim2 * dim3 * dim4 + j * dim3 * dim4 + k * dim4 + l] = array4d[i][j][k][l];
                
                }
            }
        }
    }
   
    // Print sum
     std::cout << "Sum: " << product_test <<std::endl;
   
   // Print flat vector
    // std::cout << "Flat vector: ";
    // for (int element : flat_vector) {
    //     std::cout << element << " ";
    // }
    // std::cout << std::endl;

   //OpenCL part
    std::string sourceCode;
    std::ifstream sourceFile("product.cl");
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
   std::vector<int> final_out(TOTAL_SIZE/GROUP_SIZE);
    cl::Program program(context, sourceCode);
    program.build(devices);
    cl::Buffer cl_flat_vector(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) *flat_vector.size() ,flat_vector.data());
    cl::Buffer cl_sum(context, CL_MEM_WRITE_ONLY, sizeof(int) *(TOTAL_SIZE/GROUP_SIZE));
    cl::Kernel kernel(program, "product");
    int size = dim1*dim2*dim3*dim4;
    kernel.setArg(0, cl_flat_vector);
    kernel.setArg(1, cl_sum);
    kernel.setArg(2,size );
cl::NDRange global_size(TOTAL_SIZE);
cl::NDRange local_size(GROUP_SIZE);

    cl_int err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, local_size);
    if (err != CL_SUCCESS) {
        std::cerr << "Error enqueuing kernel. Error code: " << err << std::endl;
        return {};
    }    cl::Buffer cl_sum_second(context, CL_MEM_WRITE_ONLY, sizeof(int) * (TOTAL_SIZE / GROUP_SIZE));
    kernel.setArg(0, cl_sum);
    kernel.setArg(1, cl_sum_second);
    kernel.setArg(2, final_out.size());
    cl::NDRange global_size_2(final_out.size());
    cl::NDRange local_size_2(GROUP_SIZE);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size_2,local_size_2);
    if (err != CL_SUCCESS) {
        std::cerr << "Error enqueuing kernel. Error code: " << err << std::endl;
        return {};
    }

   queue.enqueueReadBuffer(cl_sum, CL_TRUE, 0, sizeof(int) * final_out.size(),final_out.data());
   int tmp_sum=1;
   for(int i=0;i<final_out.size();i++)
   {
   // printf("%d ",final_out[i]);
    tmp_sum*=final_out[i];
   }
   printf("sum: %d",tmp_sum);

}
#include <iostream>
#include <vector>
#include <cl.hpp>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>

int main() {
    // Load input image
    cv::Mat image = cv::imread("open_cl_image.bmp", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Unable to load input image." << std::endl;
        return 1;
    }

    // Get image dimensions
    int height = image.rows;
    int width = image.cols;

    // Convert image data to vector
    std::vector<uchar> imageData;
    if (image.isContinuous()) {
        imageData.assign(image.data, image.data + image.total());
    } else {
        for (int i = 0; i < image.rows; ++i) {
            imageData.insert(imageData.end(), image.ptr<uchar>(i), image.ptr<uchar>(i) + image.cols);
        }
    }

    // Mirror the image vertically (serial implementation)
    auto start_serial = std::chrono::high_resolution_clock::now();
    std::vector<uchar> mirror_image_data(imageData.size());
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            mirror_image_data[((height - 1 - i) * width) + j] = imageData[i * width + j];
        }
    }
    auto end_serial = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_serial = end_serial - start_serial;

    // OpenCL part
    std::string sourceCode;
    std::ifstream sourceFile("ver_image.cl");
    if (!sourceFile.is_open()) {
        std::cerr << "Failed to open OpenCL source file." << std::endl;
        return 1;
    }
    sourceCode.assign(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

   
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Platform platform = platforms[0];
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    cl::Context context(devices);
    cl::CommandQueue queue(context, devices[0]);
    cl::Program program(context, sourceCode);
    program.build(devices);
    cl::Buffer cl_image_data(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uchar) * imageData.size(), imageData.data());
    cl::Buffer final_output(context, CL_MEM_WRITE_ONLY, sizeof(uchar) * imageData.size());
    cl::Kernel kernel(program, "ver_image");
    kernel.setArg(0, cl_image_data);
    kernel.setArg(1, final_output);
    kernel.setArg(2, height);
    kernel.setArg(3, width);
    std::vector<uchar> cl_image_data_host(imageData.size());

  
    auto start_opencl = std::chrono::high_resolution_clock::now();
    cl_int err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(height, width));
    if (err != CL_SUCCESS) {
        std::cerr << "Error enqueuing kernel. Error code: " << err << std::endl;
        return 1;
    }
    auto end_opencl = std::chrono::high_resolution_clock::now();
    queue.enqueueReadBuffer(final_output, CL_TRUE, 0, sizeof(uchar) * cl_image_data_host.size(), cl_image_data_host.data());
    
    std::chrono::duration<double> elapsed_opencl = end_opencl - start_opencl;


    cv::Mat resultImage(image.rows, image.cols, CV_8UC1, cl_image_data_host.data());
    cv::imwrite("output_image_mirrored.bmp", resultImage);

   
    std::cout << "Serial implementation time: " << elapsed_serial.count() << " seconds" << std::endl;
    std::cout << "OpenCL kernel execution time: " << elapsed_opencl.count() << " seconds" << std::endl;
    std::cout << "the mirrored horizontal and vertical images are stored in output_image_mirrored.bmp" << std::endl;

    return 0;
}

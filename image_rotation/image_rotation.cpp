#include <iostream>
#include <vector>
#include <cl.hpp>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("sample_cl_3.bmp", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Error: Unable to load input image." << std::endl;
        return 1;
    }

    int height = image.rows;
    int width = image.cols;
    
    std::vector<uchar> imageData;
    if (image.isContinuous()) {
        imageData.assign(image.data, image.data + image.total());
    } else {
        for (int i = 0; i < image.rows; ++i) {
            imageData.insert(imageData.end(), image.ptr<uchar>(i), image.ptr<uchar>(i) + image.cols);
        }
    }

    std::vector<uchar> mirror_image_data(imageData.size());
// image rotation cpp serial code 
for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
        mirror_image_data[(j+1)*height-i-1] = imageData[i*width +j];
    }
}


   
    std::string sourceCode;
    std::ifstream sourceFile("image_rotation.cl");
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
    cl::Buffer cl_image_data(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uchar) *imageData.size() , imageData.data());
    cl::Buffer final_output(context, CL_MEM_WRITE_ONLY, sizeof(uchar) * imageData.size());
    cl::Kernel kernel(program, "image_rotation");
    kernel.setArg(0, cl_image_data);
    kernel.setArg(1, final_output);
    kernel.setArg(2, height);
    kernel.setArg(3, width);
     std::vector<uchar> cl_image_data_host(imageData.size());
    cl_int err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(height,width));

    if (err != CL_SUCCESS) {
        std::cerr << "Error enqueuing kernel. Error code: " << err << std::endl;
        return {};
    }
    queue.enqueueReadBuffer(final_output, CL_TRUE, 0, sizeof(uchar) * cl_image_data_host.size(),cl_image_data_host.data());

    // Write mirrored image to file
    cv::Mat resultImage(image.cols, image.rows, CV_8UC1, cl_image_data_host.data());
    cv::imwrite("output_image_rotated.bmp", resultImage);
    cv::imwrite("original.bmp", image);

    std::cout << "The output of the rotated image is saved in output_image_rotated.bmp" << std::endl;

    return 0;
}

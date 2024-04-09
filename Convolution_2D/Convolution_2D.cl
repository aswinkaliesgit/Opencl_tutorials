#define INPUT_SIZE 4
__kernel void Convolution_2D(__global float* input,
                         __global float* output,
                         int input_height,
                         int input_width,
                         int output_height,
                         int output_width,
                         int kernel_size,
                         __global float* kernel_filter) {
    int i = get_global_id(0);
    int j = get_global_id(1);
     int start_point = i - (kernel_size / 2);
     int end_point = j - (kernel_size / 2);
            float sum = 0;
                        for (int k = 0; k < kernel_size; ++k) {
                for (int l = 0; l < kernel_size; ++l) {
                    if ((start_point + k >= 0 && start_point + k < INPUT_SIZE) && (end_point + l >= 0 && end_point + l < INPUT_SIZE)) {
                        sum += input[(start_point + k)*INPUT_SIZE+ end_point+l] * kernel_filter[k*kernel_size+l];
                    }
                }
                }
            output[i*output_width+j] = sum;
                       
    }
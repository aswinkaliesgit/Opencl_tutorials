#define INPUT_SIZE 10
#define DEPTH 5
__kernel void Convolution_3D(__global float* input,
                         __global float* output,
                         int input_height,
                         int input_width,
                         int output_height,
                         int output_width,
                         int kernel_size,
                         __global float* kernel_filter) {
    int c=get_global_id(0);
    int i=get_global_id(1);
    int j=get_global_id(2);


                float sum=0.0f;
                for (int m = 0; m < kernel_size; ++m) {
                    for (int n = 0; n < kernel_size; ++n) {
                        int ii = i + m - kernel_size / 2;
                        int jj = j + n - kernel_size / 2;
                        if (ii >= 0 && ii < INPUT_SIZE && jj >= 0 && jj < INPUT_SIZE) {
                            sum += kernel_filter[m * kernel_size + n] * input[(ii * INPUT_SIZE + jj) * DEPTH + c];
                        }
                    }
                }
                output[c*INPUT_SIZE*INPUT_SIZE+i*INPUT_SIZE+j]=sum;
             
            }
        

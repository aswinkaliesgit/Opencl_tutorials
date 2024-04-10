__kernel void Averagepooling(__global float* input,
                         __global float* output,
                         int input_height,
                         int input_width,
                         int output_height,
                         int output_width,
                         int kernel_size,
                         int stride) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    float local_avg = 0.0f;  // Declare local_max with float type

    for (int k = 0; k < kernel_size; k++) {
        for (int l = 0; l < kernel_size; l++) {
                 int input_index = (i * stride + k) * input_width + (j * stride + l);
                  local_avg+=input[input_index];
            
        }
    }

    int output_index = i * output_width + j;
    output[output_index] = local_avg/(kernel_size*kernel_size);
}

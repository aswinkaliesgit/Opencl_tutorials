__kernel void Maxpooling(__global float* input,
                         __global float* output,
                         int input_height,
                         int input_width,
                         int output_height,
                         int output_width,
                         int kernel_size,
                         int stride) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    float local_max = -100;  // Declare local_max with float type

    for (int k = 0; k < kernel_size; k++) {
        for (int l = 0; l < kernel_size; l++) {
                 int input_index = (i * stride + k) * input_width + (j * stride + l);
                local_max =  (local_max<input[input_index])? input[input_index]:local_max;
            
        }
    }

    int output_index = i * output_width + j;
    output[output_index] = local_max;
}

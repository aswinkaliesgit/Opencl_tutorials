__kernel void Activation_functions(__global float* input, __global float* output, int type) {
    int global_id = get_global_id(0);
    float temp;

    switch(type){
        case 1:
            // sigmoid
            temp = 1 + exp(-1 * input[global_id]);
            temp = 1 / temp;
            output[global_id] = temp;
            break;
        case 2:
            // relu
            output[global_id] = max(0.0f, input[global_id]);
            break;
        case 3:
            // tanh
            temp = 1 + exp(-2 * input[global_id]);
            temp = 2 / temp - 1;
            output[global_id] = temp;
            break;
        case 4:
            // relu 6
            output[global_id] = min(max(input[global_id], 0.0f), 6.0f);
            break;
        case 5:
            // GELU
            temp = sqrt(2 / M_PI);
            temp *= (input[global_id] + 0.044715 * pow(input[global_id], 3));
            temp = 1 + exp(-2 * temp);
            temp = 2 / temp - 1;
            temp = 0.5 * input[global_id] * (1 + temp);
            output[global_id] = temp;
            break;
        case 6:
            // SiLU
            temp = 1 + exp(-1 * input[global_id]);
            temp = 1 / temp;
            output[global_id] = temp * input[global_id];
            break;
    }
}


__kernel void naive_mul(__global float *a, __global float *b, __global float *c, int N) {
    int x_dim = get_global_id(0);
    int y_dim = get_global_id(1);
    
    if (x_dim < N && y_dim < N) {
     
        for (int k = 0; k < N; k++) {
             c[x_dim * N + y_dim]  += a[x_dim * N + k] * b[k * N + y_dim];
        }
       
    }
}


__kernel void naive_mul(__global float *a, __global float *b, __global float *c, int N) {
    int x_dim = get_global_id(0);
    float tmp;
    float a_register[1024];
    
     for(int i=0;i<N;i++)
     {
        a_register[i]=a[x_dim*N + i];
     }
     for(int l=0;l<N;l++)
     {
        tmp=0.0f;
       for (int k=0; k < N; k++) {
             tmp += a_register[k] * b[k * N +l];
        }

        c[x_dim* N + l]=tmp;
        
     }
       
    
    }
    


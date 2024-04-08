
__kernel void naive_mul(__global float *a, __global float *b, __global float *c, int N,__local float *b_local) {
    int x_dim = get_global_id(0);
    int k;
    int x_dim_local = get_local_id(0);
    int size = get_local_size(0);
    float tmp;
    float a_register[1024];
     for(int i=0;i<N;i++)
     {
        a_register[i]=a[x_dim*N + i];
     }
     for(int l=0;l<N;l++)
     {
      for(k=x_dim_local;k<N;k+=size)
      {
         b_local[k]=b[k*N +l];
      }
      
        barrier(CLK_LOCAL_MEM_FENCE);
          tmp=0.0f;
       for (k=0; k < N; k++) {
             tmp += a_register[k] * b_local[k];
        }
          
        c[x_dim* N + l] =tmp;
        

        
     }
       
    
    }
    


#define GROUP_SIZE 1204
__kernel void sum(__global int* input,__global  int* output,  int size) {
__local int partial_sums[GROUP_SIZE];
int global_id=get_global_id(0);
int local_id = get_local_id(0);

partial_sums[local_id]=input[global_id];
       barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = get_local_size(0)/ 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
           // printf("%d ",get_local_size(0));
            partial_sums[local_id] += partial_sums[local_id + stride];
        }
         barrier(CLK_LOCAL_MEM_FENCE);
        
    }
    if (local_id == 0) {
        output[get_group_id(0)] = partial_sums[0];
    }


}

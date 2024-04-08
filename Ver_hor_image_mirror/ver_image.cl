__kernel void ver_image(__global uchar * input,__global uchar*output,int height,int width )
{
    int x= get_global_id(0);
    int y = get_global_id(1);

    output[((height - 1 - x) * width) + y]=input[x*width+y];
}


__kernel void image_rotation(__global uchar * input,__global uchar*output,int height,int width )
{
    int x= get_global_id(0);
    int y = get_global_id(1);
   output[(y+1)*height-x-1] = input[x*width +y];
}


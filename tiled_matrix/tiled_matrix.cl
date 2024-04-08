
#define TILE_SIZE 16

__kernel void tiled_matrix(
    __global  float* matrixA,
    __global  float* matrixB,
    __global float* matrixC,
        int N
) 
{
    int globalRow = get_global_id(0);
    int globalCol = get_global_id(1);

    float value = 0.0f;
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];


    for (int blockK = 0; blockK < N; blockK += TILE_SIZE) {
    
        tileA[get_local_id(0)][get_local_id(1)] = matrixA[globalRow * N+ blockK + get_local_id(1)];
        tileB[get_local_id(0)][get_local_id(1)] = matrixB[(blockK + get_local_id(0)) * N + globalCol];
        barrier(CLK_LOCAL_MEM_FENCE);

    
        for (int k = 0; k < TILE_SIZE; ++k) {
            value += tileA[get_local_id(0)][k] * tileB[k][get_local_id(1)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    matrixC[globalRow * N + globalCol] = value;
}
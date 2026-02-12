#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

// Optimized transpose kernel using shared memory tiling.
// Supports large row counts by mapping rows to gridDim.x.
__global__ void transpose_kernel(const uint32_t* input, uint32_t* output, int num_rows, int num_cols) {
    __shared__ uint32_t tile[TILE_DIM][TILE_DIM + 1];

    // We map gridDim.x to num_rows (height) and gridDim.y to num_cols (width)
    // to handle num_rows > 65535.
    
    // Coordinates into the input matrix
    int x = blockIdx.y * TILE_DIM + threadIdx.x; // column index
    int y = blockIdx.x * TILE_DIM + threadIdx.y; // row index

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < num_cols && (y + j) < num_rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * num_cols + x];
        }
    }

    __syncthreads();

    // Coordinates into the output matrix (transposed)
    // The output matrix is num_cols x num_rows
    // So output index is: output_row * num_rows + output_col
    // where output_row = original_col, output_col = original_row
    
    // We swap blockIdx.x and blockIdx.y relative to input mapping
    x = blockIdx.x * TILE_DIM + threadIdx.x; // output column index (original row)
    y = blockIdx.y * TILE_DIM + threadIdx.y; // output row index (original column)

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < num_rows && (y + j) < num_cols) {
            output[(y + j) * num_rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

extern "C" void custom_transpose_outplace(
    const void* input,
    void* output,
    size_t num_rows,
    size_t num_cols,
    cudaStream_t stream
) {
    if (num_rows == 0 || num_cols == 0) return;

    // num_rows is height, num_cols is width
    // Mapping height to dimGrid.x to support > 65535 rows
    dim3 dimGrid((num_rows + TILE_DIM - 1) / TILE_DIM, (num_cols + TILE_DIM - 1) / TILE_DIM);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS);

    transpose_kernel<<<dimGrid, dimBlock, 0, stream>>>(
        (const uint32_t*)input, 
        (uint32_t*)output, 
        (int)num_rows, 
        (int)num_cols
    );
}

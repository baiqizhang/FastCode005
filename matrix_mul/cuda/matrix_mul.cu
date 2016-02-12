/*
    Copyright (C) 2011  Abhinav Jauhri (abhinav.jauhri@gmail.com), Carnegie Mellon UniversithreadIdx.y - Silicon Valley 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANthreadIdx.y; without even the implied warranthreadIdx.y of
    MERCHANTABILIthreadIdx.y or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#define TILE_WIDTH 32
#define TILE_WIDTH_SHIFT 5
#include <cuda.h>
#include <cuda_runtime.h>
#include "matrix_mul.h"
//#define TILE_WIDTH 2

namespace cuda
{
 //  __global__ 
 //  void 
 //  matrix_mul_kernel(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, int sq_dimension)
 //  {
    
 //    int threadIdx.x = threadIdx.x;
 //    int threadIdx.y = threadIdx.y;
    
 //    float sum = 0.0f;
    
 //    for(int k = 0; k < sq_dimension; k++)
 //      {
	// sum += sq_matrix_1[threadIdx.y*sq_dimension + k] * sq_matrix_2[k*sq_dimension + threadIdx.x];
 //      }
 //    sq_matrix_result[threadIdx.y*sq_dimension + threadIdx.x] = sum;
    
 //  }


  // rewrote the kernel for easier modification - Vincent
  //__global__ void matrix_mul_kernel(float *A, float *B, float *C, int d) {
   
  //  __shared__ float A_tile[blockDim.y][blockDim.x];
  //  __shared__ float B_tile[blockDim.y][blockDim.x];
    
    
  //  float temp = 0.0;
  //  for(int index = 0; index < (d/blockDim.x - 1); index++){
  //      int row = blockIdx.y * blockDim.y + threadIdx.y;
  //      int col = index * blockDim.x + threadIdx.x;
  //      *A_tile[threadIdx.y][threadIdx.x] = A[row][col];
  //      *B_tile[threadIdx.x][threadIdx.y] = B[col][row];
  //      __syncthreads();
  //      for(int k = 0; k < blockDim.x; k++){
  //          temp += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
  //      }
  //      __syncthreads();
  //  }
  //  C[blockIdx.y * blockDim.y + threadIdx.y][blockIdx.x * blockDim.x + threadIdx.x] = temp;

    // initialize row and column by index in the blocks
    //int row = blockIdx.y * blockDim.y + threadIdx.y;
    //int col = blockIdx.x * blockDim.x + threadIdx.x;

    // check if out of matrix boundary
    //if (row < d && col < d) {
    //  float tmp = 0.0;

      //loop unrolling
    //  #pragma unroll
    //  for (int i = 0; i < d; i++) {
    //    tmp += A[row * d + i] * B[i * d + col];
    //  }
    //  C[row * d + col] = tmp;
    //}
  //}
  
  // Compute C = A * B
  __global__ void matrixMultiply(float * A, float * B, float * C, int d){
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];
    int row = (blockIdx.y<<TILE_WIDTH_SHIFT) + threadIdx.y, col = (blockIdx.x<<TILE_WIDTH_SHIFT) + threadIdx.x;
    float sum = 0;

    #pragma unroll
    for (int m = 0; m < (d-1)/TILE_WIDTH+1; ++m) {
      if (row < d && (m<<TILE_WIDTH_SHIFT)+threadIdx.x < d){
        A_tile[threadIdx.y][threadIdx.x] = A[row*d + (m<<TILE_WIDTH_SHIFT)+threadIdx.x];
        //A_tile[threadIdx.y][threadIdx.x] = A[row + d*(m*TILE_WIDTH+threadIdx.x)];
      }
      else{
        A_tile[threadIdx.y][threadIdx.x] = 0;
      }
      if (col < d && (m<<TILE_WIDTH_SHIFT)+threadIdx.y < d){
        B_tile[threadIdx.y][threadIdx.x] = B[((m<<TILE_WIDTH_SHIFT)+threadIdx.y)*d+col];
        //B_tile[threadIdx.y][threadIdx.x] = B[(m*TILE_WIDTH+threadIdx.y)+d*col];
      }
      else{
        B_tile[threadIdx.y][threadIdx.x] = 0;
      }

      __syncthreads();
      #pragma unroll
      for (int k = 0; k < TILE_WIDTH; ++k)
        sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
      __syncthreads();
    }
    if (row < d && col < d)
      C[row*d + col] = sum;
  }


  void 
  matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension)
  {
    int size = sq_dimension * sq_dimension * sizeof(float);
    float *sq_matrix_1_d, *sq_matrix_2_d, *sq_matrix_result_d;
    
    /***************************************************
  1st Part: Allocation of memory on device memory  
    ****************************************************/
    
    /*
    float *A_t = (float*)calloc(sq_dimension * sq_dimension, sizeof(float));
    for (int i = 0; i < sq_dimension; i++)
        for(int j = 0; j < sq_dimension; j++){
            A_t[i * sq_dimension + j] = sq_matrix_1[j * sq_dimension + i];
        }
    */ 

    /*
    float *B_t = (float*)calloc(sq_dimension * sq_dimension, sizeof(float));
    for (int i = 0; i < sq_dimension; i++)
        for(int j = 0; j < sq_dimension; j++){
            B_t[i * sq_dimension + j] = sq_matrix_2[j * sq_dimension + i];
        }    
    */

    /* copy sq_matrix_1 and sq_matrix_2 to device memory */
    cudaMalloc((void**) &sq_matrix_1_d, size);
    cudaMemcpy(sq_matrix_1_d, sq_matrix_1, size, cudaMemcpyHostToDevice);
    cudaMalloc((void**) &sq_matrix_2_d, size);
    cudaMemcpy(sq_matrix_2_d, sq_matrix_2, size, cudaMemcpyHostToDevice);
    
    /*allocate sq_matrix_result on host */
    cudaMalloc((void**) &sq_matrix_result_d, size);
    
    /***************************************************
   2nd Part: Inovke kernel 
    ****************************************************/
    // dim3 dimBlock(sq_dimension, sq_dimension);
    // dim3 dimGrid(1,1);
    // matrix_mul_kernel<<<dimGrid, dimBlock, dimBlock.x * dimBlock.x * sizeof(float)>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);

    // increase block dimension
    //int blockDimension = 32;
    //dim3 dimBlock(blockDimension, blockDimension);
    //dim3 dimGrid(ceil(double(sq_dimension)/double(dimBlock.x)), ceil(double(sq_dimension)/double(dimBlock.y)));
    //matrix_mul_kernel<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
    dim3 dimGrid((sq_dimension-1)/TILE_WIDTH+1, (sq_dimension-1)/TILE_WIDTH+1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    matrixMultiply<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
    cudaThreadSynchronize();

    /***************************************************
   3rd Part: Transfer result from device to host 
    ****************************************************/
    cudaMemcpy(sq_matrix_result, sq_matrix_result_d, size, cudaMemcpyDeviceToHost);
    cudaFree(sq_matrix_1_d);
    cudaFree(sq_matrix_2_d);
    cudaFree(sq_matrix_result_d);
  }  
} // namespace cuda

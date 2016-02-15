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
#include "stdio.h"
//#define TILE_WIDTH 2

namespace cuda
{
  __global__ void matrixMultiply_1000(float * A, float * B, float * C, int d){
    __shared__ float A_tile[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[2][TILE_WIDTH][TILE_WIDTH];
    int row = (blockIdx.y<<TILE_WIDTH_SHIFT) + threadIdx.y, col = (blockIdx.x<<TILE_WIDTH_SHIFT) + threadIdx.x;
    float sum = 0;
    
    #pragma unroll
    for (int m = 0; m < 30; m+=2) {
      A_tile[0][threadIdx.y][threadIdx.x] = A[row*d + (m<<TILE_WIDTH_SHIFT)+threadIdx.x];
      B_tile[0][threadIdx.y][threadIdx.x] = B[((m<<TILE_WIDTH_SHIFT)+threadIdx.y)*d+col];


      A_tile[1][threadIdx.y][threadIdx.x] = A[row*d + ((m+1)<<TILE_WIDTH_SHIFT)+threadIdx.x];
      B_tile[1][threadIdx.y][threadIdx.x] = B[(((m+1)<<TILE_WIDTH_SHIFT)+threadIdx.y)*d+col];

      __syncthreads();
      #pragma unroll
      for (int k = 0; k < TILE_WIDTH; ++k){
        sum += A_tile[0][threadIdx.y][k] * B_tile[0][k][threadIdx.x];
        sum += A_tile[1][threadIdx.y][k] * B_tile[1][k][threadIdx.x];
      }
      __syncthreads();
    }
    
    #pragma unroll
    for (int m = 30; m < 32; m+=2) {
      if ((m<<TILE_WIDTH_SHIFT)+threadIdx.x < d)
        A_tile[0][threadIdx.y][threadIdx.x] = A[row*d + (m<<TILE_WIDTH_SHIFT)+threadIdx.x];
      else
        A_tile[0][threadIdx.y][threadIdx.x] = 0;

      if ((m<<TILE_WIDTH_SHIFT)+threadIdx.y < d)
        B_tile[0][threadIdx.y][threadIdx.x] = B[((m<<TILE_WIDTH_SHIFT)+threadIdx.y)*d+col];
      else
        B_tile[0][threadIdx.y][threadIdx.x] = 0;

      if (((m+1)<<TILE_WIDTH_SHIFT)+threadIdx.x < d)
        A_tile[1][threadIdx.y][threadIdx.x] = A[row*d + ((m+1)<<TILE_WIDTH_SHIFT)+threadIdx.x];
      else
        A_tile[1][threadIdx.y][threadIdx.x] = 0;

      if (((m+1)<<TILE_WIDTH_SHIFT)+threadIdx.y < d)
        B_tile[1][threadIdx.y][threadIdx.x] = B[(((m+1)<<TILE_WIDTH_SHIFT)+threadIdx.y)*d+col];
      else
        B_tile[1][threadIdx.y][threadIdx.x] = 0;

      __syncthreads();
      #pragma unroll
      for (int k = 0; k < TILE_WIDTH; ++k){
        sum += A_tile[0][threadIdx.y][k] * B_tile[0][k][threadIdx.x];
        sum += A_tile[1][threadIdx.y][k] * B_tile[1][k][threadIdx.x];
      }
      __syncthreads();
    }
    if (row < d && col < d)
      C[row*d + col] = sum;
  }

  __global__ void matrixMultiply_1000_2(float * A, float * B, float * C, int d){
    __shared__ float A_tile[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[2][TILE_WIDTH][TILE_WIDTH];
    int row = (blockIdx.y<<TILE_WIDTH_SHIFT) + threadIdx.y, col = (blockIdx.x<<TILE_WIDTH_SHIFT) + threadIdx.x;
    float sum = 0;
    
    #pragma unroll
    for (int m = 0; m < 30; m+=2) {
      A_tile[0][threadIdx.y][threadIdx.x] = A[row*d + (m<<TILE_WIDTH_SHIFT)+threadIdx.x];
      B_tile[0][threadIdx.y][threadIdx.x] = B[((m<<TILE_WIDTH_SHIFT)+threadIdx.y)*d+col];

      if (m!=0){
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k){
          sum += A_tile[1][threadIdx.y][k] * B_tile[1][k][threadIdx.x];
        }
      }
      __syncthreads();
      
      #pragma unroll
      for (int k = 0; k < TILE_WIDTH; ++k){
        sum += A_tile[0][threadIdx.y][k] * B_tile[0][k][threadIdx.x];
      }
      
      A_tile[1][threadIdx.y][threadIdx.x] = A[row*d + ((m+1)<<TILE_WIDTH_SHIFT)+threadIdx.x];
      B_tile[1][threadIdx.y][threadIdx.x] = B[(((m+1)<<TILE_WIDTH_SHIFT)+threadIdx.y)*d+col];

      __syncthreads();
    }
    
    #pragma unroll
    for (int m = 30; m < 32; m+=2) {
      if ((m<<TILE_WIDTH_SHIFT)+threadIdx.x < d)
        A_tile[0][threadIdx.y][threadIdx.x] = A[row*d + (m<<TILE_WIDTH_SHIFT)+threadIdx.x];
      else
        A_tile[0][threadIdx.y][threadIdx.x] = 0;

      if ((m<<TILE_WIDTH_SHIFT)+threadIdx.y < d)
        B_tile[0][threadIdx.y][threadIdx.x] = B[((m<<TILE_WIDTH_SHIFT)+threadIdx.y)*d+col];
      else
        B_tile[0][threadIdx.y][threadIdx.x] = 0;

      #pragma unroll
      for (int k = 0; k < TILE_WIDTH; ++k){
        sum += A_tile[1][threadIdx.y][k] * B_tile[1][k][threadIdx.x];
      }

      __syncthreads();
      #pragma unroll
      for (int k = 0; k < TILE_WIDTH; ++k){
        sum += A_tile[0][threadIdx.y][k] * B_tile[0][k][threadIdx.x];
      }
      if (((m+1)<<TILE_WIDTH_SHIFT)+threadIdx.x < d)
        A_tile[1][threadIdx.y][threadIdx.x] = A[row*d + ((m+1)<<TILE_WIDTH_SHIFT)+threadIdx.x];
      else
        A_tile[1][threadIdx.y][threadIdx.x] = 0;

      if (((m+1)<<TILE_WIDTH_SHIFT)+threadIdx.y < d)
        B_tile[1][threadIdx.y][threadIdx.x] = B[(((m+1)<<TILE_WIDTH_SHIFT)+threadIdx.y)*d+col];
      else
        B_tile[1][threadIdx.y][threadIdx.x] = 0;
      __syncthreads();
    }
    #pragma unroll
    for (int k = 0; k < TILE_WIDTH; ++k){
        sum += A_tile[1][threadIdx.y][k] * B_tile[1][k][threadIdx.x];
    }
    if (row < d && col < d)
      C[row*d + col] = sum;
  }

  __global__ void matrixMultiply_1024(float * A, float * B, float * C, int d){
    __shared__ float A_tile[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[2][TILE_WIDTH][TILE_WIDTH];
    int row = (blockIdx.y<<TILE_WIDTH_SHIFT) + threadIdx.y, col = (blockIdx.x<<TILE_WIDTH_SHIFT) + threadIdx.x;
    float sum = 0;
    
    #pragma unroll
    for (int m = 0; m < 32; m+=2) {
      if (m!=0){
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k){
          sum += A_tile[1][threadIdx.y][k] * B_tile[1][k][threadIdx.x];
          //sum += A_tile[1][threadIdx.y][k] * B_tile[1][k][threadIdx.x];
        }
      }

      A_tile[0][threadIdx.y][threadIdx.x] = A[row*d + (m<<TILE_WIDTH_SHIFT)+threadIdx.x];
      B_tile[0][threadIdx.y][threadIdx.x] = B[((m<<TILE_WIDTH_SHIFT)+threadIdx.y)*d+col];


      __syncthreads();
      #pragma unroll
      for (int k = 0; k < TILE_WIDTH; ++k){
        sum += A_tile[0][threadIdx.y][k] * B_tile[0][k][threadIdx.x];
        //sum += A_tile[1][threadIdx.y][k] * B_tile[1][k][threadIdx.x];
      }

      A_tile[1][threadIdx.y][threadIdx.x] = A[row*d + ((m+1)<<TILE_WIDTH_SHIFT)+threadIdx.x];
      B_tile[1][threadIdx.y][threadIdx.x] = B[(((m+1)<<TILE_WIDTH_SHIFT)+threadIdx.y)*d+col];
      __syncthreads();
    }
    
    #pragma unroll
    for (int k = 0; k < TILE_WIDTH; ++k){
        sum += A_tile[1][threadIdx.y][k] * B_tile[1][k][threadIdx.x];
    }

    if (row < d && col < d)
      C[row*d + col] = sum;
  }
  
  __global__ void matrixMultiply_1024_2(float * A, float * B, float * C, int d){
    __shared__ float A_tile[4][TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[4][TILE_WIDTH][TILE_WIDTH];
    int row = (blockIdx.y<<TILE_WIDTH_SHIFT) + threadIdx.y, col = (blockIdx.x<<TILE_WIDTH_SHIFT) + threadIdx.x;
    float sum = 0,sum2=0;
    
    #pragma unroll
    for (int m = 0; m < 32; m+=4) {
      if (m!=0){
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k){
          sum += A_tile[2][threadIdx.y][k] * B_tile[2][k][threadIdx.x];
          sum2 += A_tile[3][threadIdx.y][k] * B_tile[3][k][threadIdx.x];
        }
      }

      A_tile[0][threadIdx.y][threadIdx.x] = A[row*d + (m<<TILE_WIDTH_SHIFT)+threadIdx.x];
      B_tile[0][threadIdx.y][threadIdx.x] = B[((m<<TILE_WIDTH_SHIFT)+threadIdx.y)*d+col];
      A_tile[1][threadIdx.y][threadIdx.x] = A[row*d + ((m+1)<<TILE_WIDTH_SHIFT)+threadIdx.x];
      B_tile[1][threadIdx.y][threadIdx.x] = B[(((m+1)<<TILE_WIDTH_SHIFT)+threadIdx.y)*d+col];


      __syncthreads();
      #pragma unroll
      for (int k = 0; k < TILE_WIDTH; ++k){
        sum += A_tile[0][threadIdx.y][k] * B_tile[0][k][threadIdx.x];
        sum2 += A_tile[1][threadIdx.y][k] * B_tile[1][k][threadIdx.x];
      }

      A_tile[2][threadIdx.y][threadIdx.x] = A[row*d + ((m+2)<<TILE_WIDTH_SHIFT)+threadIdx.x];
      B_tile[2][threadIdx.y][threadIdx.x] = B[(((m+2)<<TILE_WIDTH_SHIFT)+threadIdx.y)*d+col];
      A_tile[3][threadIdx.y][threadIdx.x] = A[row*d + ((m+3)<<TILE_WIDTH_SHIFT)+threadIdx.x];
      B_tile[3][threadIdx.y][threadIdx.x] = B[(((m+3)<<TILE_WIDTH_SHIFT)+threadIdx.y)*d+col];
      __syncthreads();
    }
    
    #pragma unroll
    for (int k = 0; k < TILE_WIDTH; ++k){
          sum += A_tile[2][threadIdx.y][k] * B_tile[2][k][threadIdx.x];
          sum2 += A_tile[3][threadIdx.y][k] * B_tile[3][k][threadIdx.x];
    }

    if (row < d && col < d)
      C[row*d + col] = sum+sum2;
  }

  // Compute C = A * B
  __global__ void matrixMultiply(float * A, float * B, float * C, int d){
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];
    int row = (blockIdx.y<<TILE_WIDTH_SHIFT) + threadIdx.y, col = (blockIdx.x<<TILE_WIDTH_SHIFT) + threadIdx.x;
    float sum = 0;
    
    #pragma unroll
    for (int m = 0; m < (d-1)/TILE_WIDTH+1; ++m) {
      if ((m<<TILE_WIDTH_SHIFT)+threadIdx.x < d)
        A_tile[threadIdx.y][threadIdx.x] = A[row*d + (m<<TILE_WIDTH_SHIFT)+threadIdx.x];
      else
        A_tile[threadIdx.y][threadIdx.x] = 0;

      if ((m<<TILE_WIDTH_SHIFT)+threadIdx.y < d)
        B_tile[threadIdx.y][threadIdx.x] = B[((m<<TILE_WIDTH_SHIFT)+threadIdx.y)*d+col];
      else
        B_tile[threadIdx.y][threadIdx.x] = 0;

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
    dim3 dimGrid((sq_dimension-1)/TILE_WIDTH+1, (sq_dimension-1)/TILE_WIDTH+1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    if (sq_dimension==1024){
      matrixMultiply_1024_2<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
    }else if (sq_dimension == 1000){
      matrixMultiply_1000_2<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
    }else{
      matrixMultiply<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
    }
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


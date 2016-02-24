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
#include <sys/time.h>
//#define OUTPUT_TIME 
//#define TILE_WIDTH 2

namespace cuda
{

__device__ void saxpy( float a, float *b, float *c )
{
    c[0] += a*b[0];
    c[1] += a*b[1];
    c[2] += a*b[2];
    c[3] += a*b[3];
    c[4] += a*b[4];
    c[5] += a*b[5];
    c[6] += a*b[6];
    c[7] += a*b[7];
    c[8] += a*b[8];
    c[9] += a*b[9];
    c[10] += a*b[10];
    c[11] += a*b[11];
    c[12] += a*b[12];
    c[13] += a*b[13];
    c[14] += a*b[14];
    c[15] += a*b[15];
    c[16] += a*b[16];
    c[17] += a*b[17];
    c[18] += a*b[18];
    c[19] += a*b[19];
    c[20] += a*b[20];
    c[21] += a*b[21];
    c[22] += a*b[22];
    c[23] += a*b[23];
    c[24] += a*b[24];
    c[25] += a*b[25];
    c[26] += a*b[26];
    c[27] += a*b[27];
    c[28] += a*b[28];
    c[29] += a*b[29];
    c[30] += a*b[30];
    c[31] += a*b[31];
}

#define STEP 32
#define GROUP 4
//thread : (STEP,GROUP)

 __global__ void matrixMultiply_1024( const float *A, const float *B, float* C)
{
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x * STEP * GROUP;
    const int iby = blockIdx.y * STEP;
    const int id = inx + iny * STEP;

    A += ibx + id;
    B += inx + ( iby + iny) * 1024 ;
    C += ibx + id  + ( iby * 1024 );
    
//    if (blockIdx.x!=2||blockIdx.y!=3)
//        return;
//    printf("(%d,%d:%d,%d):A+=%d B+=%d C+=%d \n", blockIdx.x , blockIdx.y, inx,iny,
//             ibx + id, inx + ( iby + iny) * ldb ,ibx + id  + ( iby * ldc ));
    
    //const float *Blast = B + dim;

    float c[STEP] = {0};

    __shared__ float bs[STEP][STEP + 1];
    //do
#pragma unroll
    for (int t=0;t<1024/STEP;t++)
    {
//#pragma unroll
        //for( int i = 0; i < STEP; i += GROUP*2 ){
        //    bs[inx][iny+i]  = B[i*dim];
        //}
            bs[inx][iny]  = B[0];
            bs[inx][iny+GROUP]  = B[(GROUP)*1024];
            bs[inx][iny+2*GROUP]  = B[(2*GROUP)*1024];
            bs[inx][iny+3*GROUP]  = B[(3*GROUP)*1024];
            bs[inx][iny+4*GROUP]  = B[(4*GROUP)*1024];
            bs[inx][iny+5*GROUP]  = B[(5*GROUP)*1024];
            bs[inx][iny+6*GROUP]  = B[(6*GROUP)*1024];
            bs[inx][iny+7*GROUP]  = B[(7*GROUP)*1024];


        __syncthreads();

#pragma unroll
        for( int i = 0; i < STEP; i++, A += 1024 ){
            saxpy( A[0], &bs[i][0], c ); 
        }
        
        B += STEP;

        
        __syncthreads();
    } //while( B < Blast );

    for( int i = 0; i < STEP; i++, C += 1024 )
        C[0] = c[i]; 
}	



 __global__ void matrixMultiply_1000( const float *A, const float *B, float* C)
{
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x * STEP * GROUP;
    const int iby = blockIdx.y * STEP;
    const int id = inx + iny * STEP;

    A += ibx + id;
    B += inx + ( iby + iny) * 1000 ;
    C += ibx + id  + ( iby * 1000 );
    
    //printf("[%d,%d] ",blockIdx.x,blockIdx.y);

    float c[STEP] = {0};

    __shared__ float bs[STEP][STEP + 1];
    
//    if (threadIdx.x==0 && threadIdx.y==0)
//                printf("ix:%d iy:%d \n",inx,iny);
    //do
#pragma unroll
    for (int t=0;t<(1000-1)/STEP+1;t++)
    {
#pragma unroll
        for( int i = 0; i < STEP; i += GROUP ){
            if ((t*STEP+inx>=1000) || (iby+iny+i>=1000))
                bs[inx][iny+i] = 0;
            else {
                bs[inx][iny+i] = B[i*1000];
  //              printf("i:%d ix:%d iy:%d \n",i,inx,iny);
            }
        }


        __syncthreads();

  if (ibx+id<1000){
#pragma unroll
        for( int i = 0; i < STEP; i++, A += 1000 ){
            if (t*STEP+i<1000) {
                saxpy( A[0], &bs[i][0], c ); 
                //printf("i:%d %f * %f,%f\n",i,A[0],bs[i][0],bs[i][1]);
            }
        }
  }
        B += STEP;

        
        __syncthreads();
    } //while( B < Blast );


  if (ibx+id<1000){
    for( int i = 0; i < STEP; i++, C += 1000 )
        if (iby+i<1000){
            C[0] = c[i]; 
            //printf("C[0] = %f ",C[0]);
        }
  }

}	



 __global__ void matrixMultiply( const float *A, const float *B, float* C, int dim )
{
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x * STEP * GROUP;
    const int iby = blockIdx.y * STEP;
    const int id = inx + iny * STEP;

    A += ibx + id;
    B += inx + ( iby + iny) * dim ;
    C += ibx + id  + ( iby * dim );
    
    //printf("[%d,%d] ",blockIdx.x,blockIdx.y);

    float c[STEP] = {0};

    __shared__ float bs[STEP][STEP + 1];
    
//    if (threadIdx.x==0 && threadIdx.y==0)
//                printf("ix:%d iy:%d \n",inx,iny);
    //do
#pragma unroll
    for (int t=0;t<(dim-1)/STEP+1;t++)
    {
#pragma unroll
        for( int i = 0; i < STEP; i += GROUP ){
            if ((t*STEP+inx>=dim) || (iby+iny+i>=dim))
                bs[inx][iny+i] = 0;
            else {
                bs[inx][iny+i] = B[i*dim];
  //              printf("i:%d ix:%d iy:%d \n",i,inx,iny);
            }
        }


        __syncthreads();

  if (ibx+id<dim){
#pragma unroll
        for( int i = 0; i < STEP; i++, A += dim ){
            if (t*STEP+i<dim) {
                saxpy( A[0], &bs[i][0], c ); 
                //printf("i:%d %f * %f,%f\n",i,A[0],bs[i][0],bs[i][1]);
            }
        }
  }
        B += STEP;

        
        __syncthreads();
    } //while( B < Blast );


  if (ibx+id<dim){
    for( int i = 0; i < STEP; i++, C += dim )
        if (iby+i<dim){
            C[0] = c[i]; 
            //printf("C[0] = %f ",C[0]);
        }
  }

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

#ifdef OUTPUT_TIME
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);
#endif

    //printf("\n\ndim=%d",sq_dimension);
    dim3 grid( (sq_dimension-1)/STEP/GROUP+1, (sq_dimension-1)/STEP+1 ), 
         threads(STEP, GROUP);
    if (sq_dimension==1024){
        matrixMultiply_1024<<<grid, threads>>>( sq_matrix_2_d, sq_matrix_1_d, sq_matrix_result_d);
    }else if (sq_dimension == 1000){
        matrixMultiply_1000<<<grid, threads>>>( sq_matrix_2_d, sq_matrix_1_d, sq_matrix_result_d);
    } else {
        /*
        printf("\nA = \n");
        for (int i=0;i<sq_dimension;i++){
           for(int j=0;j<sq_dimension;j++)
               printf("%f ",sq_matrix_1[i*sq_dimension+j]);
           printf("\n");
        }

        printf("\nB = \n");
        for (int i=0;i<sq_dimension;i++){
           for(int j=0;j<sq_dimension;j++)
               printf("%f ",sq_matrix_2[i*sq_dimension+j]);
           printf("\n");
        }*/
      matrixMultiply<<<grid, threads>>>(sq_matrix_2_d, sq_matrix_1_d, sq_matrix_result_d, sq_dimension);
    }
//    }else{
//      matrixMultiply<<<dimGrid, dimBlock>>>(sq_matrix_1_d, sq_matrix_2_d, sq_matrix_result_d, sq_dimension);
//    }
    cudaThreadSynchronize();

#ifdef OUTPUT_TIME
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    printf(" %ld.%06ld\t", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
#endif

    /***************************************************
    3rd Part: Transfer result from device to host
    ****************************************************/
    cudaMemcpy(sq_matrix_result, sq_matrix_result_d, size, cudaMemcpyDeviceToHost);
    cudaFree(sq_matrix_1_d);
    cudaFree(sq_matrix_2_d);
    cudaFree(sq_matrix_result_d);
       
    //if (sq_dimension==4){
    /*
        printf("\nC = \n");
        for (int i=0;i<sq_dimension;i++){
            for(int j=0;j<sq_dimension;j++){
                printf("%f:",sq_matrix_result[i*sq_dimension+j]);
                float sum = 0;
                for (int k=0;k<sq_dimension;k++)
                    sum+=sq_matrix_1[i*sq_dimension+k]*sq_matrix_2[k*sq_dimension+j];
                printf("%f ",sum);
            }
            printf("\n");
        }
//    }
*/
  }
} // namespace cuda




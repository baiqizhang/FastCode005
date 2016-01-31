/*

    Copyright (C) 2011  Abhinav Jauhri (abhinav.jauhri@gmail.com), Carnegie Mellon University - Silicon Valley 

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <omp.h>
#include "matrix_mul.h"
#include <stdlib.h>
#include <stdio.h>
#include <pmmintrin.h>


namespace omp
{
    void
    matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension )
    {
        //Test Case 5	68.302 milliseconds
        float *A, *B_t;
        unsigned int n = sq_dimension, N;
        unsigned int i, j, k;
        
        // if n is not multiple of 4, create padding. N = n + 4 - (n&3). O(N^2)
        if ((n & 3) != 0){
            N = n - (n&3) + 4;
            A = (float*)calloc(N*N, sizeof(float)); // filled with 0
        } else {
            N = n;
            A = sq_matrix_1;
        }
        
        //transpose B
        B_t = (float*)calloc(N*N, sizeof(float));
        for (i = 0; i < n; i++)
            for(j = 0; j < n; j++){
                if (N != n) //n is not mul of 4, fill A
                    A[i * N + j] = sq_matrix_1[i * n + j];
                B_t[i * N + j] = sq_matrix_2[j * n + i];
            }
        
        float temp[8], result;

#pragma omp parallel for \
    private(i,j,temp,result) \
    shared(sq_matrix_result,A,B_t)
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++) {
                // SIMD
                __m128 sum = _mm_setzero_ps();
                
                // mul and sum 4 pairs of float in 4 instructions
                for (k = 0; k < n; k += 4) {
                    sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load_ps(&A[i * N + k]),
                                                     _mm_load_ps(&B_t[j * N + k])));
                }
                // store __m128 to float array, sum up and save
                _mm_store_ps(temp, sum);
                result = temp[0] + temp[1] + temp[2] + temp[3];
                sq_matrix_result[i*n + j] = result;
            }
        
        //free
        if (n != N)
            free(A);
        free(B_t);
    }
}

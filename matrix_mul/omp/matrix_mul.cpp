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

namespace omp
{
    void
    matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2, float *sq_matrix_result, unsigned int sq_dimension )
    {
        omp_set_num_threads(8);
        unsigned int i, j, k;
        
#pragma omp parallel for
        for (i = 0; i < sq_dimension; i++)
            for(j = 0; j < sq_dimension; j++)
                sq_matrix_result[i*sq_dimension + j] = 0;
        
        unsigned int N = sq_dimension, jj, kk, s = 125;
        
# pragma omp parallel for \
  shared(sq_matrix_1,sq_matrix_2,sq_matrix_result) private(i,j,k,kk,jj) \
  schedule(static)
        for(kk = 0 ;kk < N; kk += s)
            for(jj = 0;jj < N;jj += s)
                for(i=0;i<N;i++)
                    for(k = kk; k<((kk+s)>N?N:(kk+s)); k++)
                        for(j = jj; j<((jj+s)>N?N:(jj+s)); j++)
                            sq_matrix_result[i*sq_dimension+j] +=
                                sq_matrix_1[i*sq_dimension+k]*sq_matrix_2[k*sq_dimension+j];
/*
//#pragma omp parallel for
        for (i = 0; i < sq_dimension; i++)
            for (k = 0; k < sq_dimension; k++)
                for(j = 0; j < sq_dimension; j++)
	                sq_matrix_result[i*sq_dimension + j] +=
                        sq_matrix_1[i*sq_dimension + k] * sq_matrix_2[k*sq_dimension + j];
*/
    }
}

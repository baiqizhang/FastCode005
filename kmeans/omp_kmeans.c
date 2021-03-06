/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         kmeans_clustering.c  (OpenMP version)                     */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>

#include <omp.h>
#include "kmeans.h"
#include "math.h"
#include <pmmintrin.h>



/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__inline static
float euclid_dist_2(int    numdims,  /* no. dimensions */
                    float *coord1,   /* [numdims] */
                    float *coord2)   /* [numdims] */
{
    int i=0;
    float ans=0.0;
    //    float temp[8]={0};
    
    for (i=0; i<numdims-3; i+=4){
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);
        ans += (coord1[i+1]-coord2[i+1]) * (coord1[i+1]-coord2[i+1]);
        ans += (coord1[i+2]-coord2[i+2]) * (coord1[i+2]-coord2[i+2]);
        ans += (coord1[i+3]-coord2[i+3]) * (coord1[i+3]-coord2[i+3]);
    }
    for (; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);
    return(ans);
}
__inline static
float euclid_dist_8(int    numdims,  /* no. dimensions */
                    float *coord1,   /* [numdims] */
                    float *coord2)   /* [numdims] */
{
    float ans=0.0;
    
    ans += (coord1[0]-coord2[0]) * (coord1[0]-coord2[0]);
    ans += (coord1[1]-coord2[1]) * (coord1[1]-coord2[1]);
    ans += (coord1[2]-coord2[2]) * (coord1[2]-coord2[2]);
    ans += (coord1[3]-coord2[3]) * (coord1[3]-coord2[3]);
    ans += (coord1[4]-coord2[4]) * (coord1[4]-coord2[4]);
    ans += (coord1[5]-coord2[5]) * (coord1[5]-coord2[5]);
    ans += (coord1[6]-coord2[6]) * (coord1[6]-coord2[6]);
    ans += (coord1[7]-coord2[7]) * (coord1[7]-coord2[7]);

    return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__inline static
int find_nearest_cluster(int     numClusters, /* no. clusters */
                         int     numCoords,   /* no. coordinates */
                         float  *object,      /* [numCoords] */
                         float **clusters)    /* [numClusters][numCoords] */
{
    int   index, i, j;
    float dist, min_dist;
    
    /* find the cluster id that has min distance to object */
    index    = 0;
    min_dist = euclid_dist_2(numCoords, object, clusters[0]);
    
    dist = euclid_dist_2(numCoords, object, clusters[1]);
    if (dist < min_dist) { /* find the min and its array index */
        min_dist = dist;
        index    = 1;
    }
    dist = euclid_dist_2(numCoords, object, clusters[2]);
    if (dist < min_dist) { /* find the min and its array index */
        min_dist = dist;
        index    = 2;
    }
    return(index);
}


/*----< find_nearest_cluster() >---------------------------------------------*/
__inline static
int find_nearest_cluster_8(int     numClusters, /* no. clusters */
                         int     numCoords,   /* no. coordinates */
                         float  *object,      /* [numCoords] */
                         float **clusters)    /* [numClusters][numCoords] */
{
    int   index, i, j;
    float dist, min_dist;
    
    /* find the cluster id that has min distance to object */
    index    = 0;
    min_dist = euclid_dist_8(numCoords, object, clusters[0]);
    
    dist = euclid_dist_8(numCoords, object, clusters[1]);
    if (dist < min_dist) { /* find the min and its array index */
        min_dist = dist;
        index    = 1;
    }
    dist = euclid_dist_8(numCoords, object, clusters[2]);
    if (dist < min_dist) { /* find the min and its array index */
        min_dist = dist;
        index    = 2;
    }
    return(index);
}



/*----< kmeans_clustering() >------------------------------------------------*/
/* return an array of cluster centers of size [numClusters][numCoords]       */
float** omp_kmeans(int     is_perform_atomic, /* in: */
                   float **objects,           /* in: [numObjs][numCoords] */
                   int     numCoords,         /* no. coordinates */
                   int     numObjs,           /* no. objects */
                   int     numClusters,       /* no. clusters */
                   float   threshold,         /* % objects change membership */
                   int    *membership)        /* out: [numObjs] */
{
    
    int      i, j, k, index, loop=0;
    int ilim = numObjs-(numObjs&7);
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                              new cluster */
    float    delta;          /* % of objects change their clusters */
    int delta2;
    float  **clusters;       /* out: [numClusters][numCoords] */
    float  **newClusters;    /* [numClusters][numCoords] */
    double   timing;
    
    int      nthreads;             /* no. threads */
    int    **local_newClusterSize; /* [nthreads][numClusters] */
    float ***local_newClusters;    /* [nthreads][numClusters][numCoords] */
    
    // disable atomic
    is_perform_atomic = 0;
    
    nthreads = omp_get_max_threads();
    
    /* allocate a 2D space for returning variable clusters[] (coordinates
     of cluster centers) */
    clusters    = (float**) malloc(numClusters *             sizeof(float*));
    assert(clusters != NULL);
    clusters[0] = (float*)  malloc(numClusters * numCoords * sizeof(float));
    assert(clusters[0] != NULL);
    
    // loop unrolling - added by Vincent
    for (i = 1; i < numClusters-3; i += 4) {
        clusters[i] = clusters[i-1] + numCoords;
        clusters[i+1] = clusters[i] + numCoords;
        clusters[i+2] = clusters[i+1] + numCoords;
        clusters[i+3] = clusters[i+2] + numCoords;
    }
    for (j = i; j < numClusters; j++) clusters[j] = clusters[j-1] + numCoords;
    
    for (i=0; i<numClusters; i++)
        for (j=0; j<numCoords; j++)
            clusters[i][j] = objects[i][j];
    
    /* initialize membership[] */
    // loop unrolling - added by Vincent

#pragma omp parallel for shared(membership)
    for (i=0; i<numObjs-3; i += 8) {
        membership[i] = -1;
        membership[i+1] = -1;
        membership[i+2] = -1;
        membership[i+3] = -1;
        membership[i+4] = -1;
        membership[i+5] = -1;
        membership[i+6] = -1;
        membership[i+7] = -1;
    }
    for (j = i; j < numObjs; j++) membership[j] = -1;
    
    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);
    
    newClusters    = (float**) malloc(numClusters *            sizeof(float*));
    assert(newClusters != NULL);
    newClusters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
    assert(newClusters[0] != NULL);
    
    // loop unrolling - added by Vincent
    for (i=1; i < numClusters-3; i += 4) {
        newClusters[i] = newClusters[i-1] + numCoords;
        newClusters[i+1] = newClusters[i] + numCoords;
        newClusters[i+2] = newClusters[i+1] + numCoords;
        newClusters[i+3] = newClusters[i+2] + numCoords;
    }
    for (j = i; j < numClusters; j++) newClusters[j] = newClusters[j-1] + numCoords;
    
    if (!is_perform_atomic) {
        /* each thread calculates new centers using a private space,
         then thread 0 does an array reduction on them. This approach
         should be faster */
        local_newClusterSize    = (int**) malloc(nthreads * sizeof(int*));
        assert(local_newClusterSize != NULL);
        local_newClusterSize[0] = (int*)  calloc(nthreads*numClusters,
                                                 sizeof(int));
        assert(local_newClusterSize[0] != NULL);
        for (i=1; i<nthreads; i++)
            local_newClusterSize[i] = local_newClusterSize[i-1]+numClusters;
        
        /* local_newClusters is a 3D array */
        local_newClusters    =(float***)malloc(nthreads * sizeof(float**));
        assert(local_newClusters != NULL);
        local_newClusters[0] =(float**) malloc(nthreads * numClusters *
                                               sizeof(float*));
        assert(local_newClusters[0] != NULL);
        for (i=1; i<nthreads; i++)
            local_newClusters[i] = local_newClusters[i-1] + numClusters;
        for (i=0; i<nthreads; i++) {
            for (j=0; j<numClusters; j++) {
                local_newClusters[i][j] = (float*)calloc(numCoords,
                                                         sizeof(float));
                assert(local_newClusters[i][j] != NULL);
            }
        }
    }
    
    //    int deltas[20] = {0};
    
    if (_debug) timing = omp_get_wtime();
    do {
        delta = 0.0;
        delta2 = 0;
        //        for (int kk = 0 ; kk<20;kk++)
        //            deltas[kk]=0;
        
        if (is_perform_atomic) {
#pragma omp parallel for \
private(i,j,index) \
firstprivate(numObjs,numClusters,numCoords) \
shared(objects,clusters,membership,newClusters,newClusterSize) \
schedule(static) \
reduction(+:delta)
            for (i=0; i<numObjs; i++) {
                /* find the array index of nestest cluster center */
                index = find_nearest_cluster(numClusters, numCoords, objects[i],
                                             clusters);
                
                /* if membership changes, increase delta by 1 */
                if (membership[i] != index) delta += 1.0;
                
                /* assign the membership to object i */
                membership[i] = index;
                
                /* update new cluster centers : sum of objects located within */
#pragma omp atomic
                newClusterSize[index]++;
                for (j=0; j<numCoords; j++)
#pragma omp atomic
                    newClusters[index][j] += objects[i][j];
            }
        }
        else {
            if (numCoords == 8 ){
                
                
#pragma omp parallel \
shared(objects,clusters,membership,local_newClusters,local_newClusterSize)
                {
                    int tid = omp_get_thread_num();
#pragma omp for \
private(i,j) \
firstprivate(numObjs,numClusters,numCoords) \
schedule(static) \
reduction(+:delta2)
                    //firstprivate: Listed variables are initialized according to the value of their original objects prior to entry into the parallel or work-sharing construct.
                    for (i=0; i<ilim; i+=8) {
                        /* find the array index of nestest cluster center */
                        //n*k*d
                        int index1 = find_nearest_cluster_8(numClusters, numCoords,
                                                          objects[i], clusters);
                        
                        int index2 = find_nearest_cluster_8(numClusters, numCoords,
                                                          objects[i+1], clusters);
                        int index3 = find_nearest_cluster_8(numClusters, numCoords,
                                                          objects[i+2], clusters);
                        
                        int index4 = find_nearest_cluster_8(numClusters, numCoords,
                                                          objects[i+3], clusters);
                        
                        int index5 = find_nearest_cluster_8(numClusters, numCoords,
                                                            objects[i+4], clusters);
                        int index6 = find_nearest_cluster_8(numClusters, numCoords,
                                                            objects[i+5], clusters);
                        
                        int index7 = find_nearest_cluster_8(numClusters, numCoords,
                                                            objects[i+6], clusters);
                        
                        int index8 = find_nearest_cluster_8(numClusters, numCoords,
                                                            objects[i+7], clusters);
                        
                        if (membership[i] != index1) delta2 ++;
                        if (membership[i+1] != index2) delta2 ++;
                        if (membership[i+2] != index3) delta2 ++;
                        if (membership[i+3] != index4) delta2 ++;

                        if (membership[i+4] != index5) delta2 ++;
                        if (membership[i+5] != index6) delta2 ++;
                        if (membership[i+6] != index7) delta2 ++;
                        if (membership[i+7] != index8) delta2 ++;
                        
                        /* assign the membership to object i */
                        membership[i] = index1;
                        membership[i+1] = index2;
                        membership[i+2] = index3;
                        membership[i+3] = index4;
                        
                        membership[i+4] = index5;
                        membership[i+5] = index6;
                        membership[i+6] = index7;
                        membership[i+7] = index8;

                        
                        /* update new cluster centers : sum of all objects located
                         within (average will be performed later) */
                        local_newClusterSize[tid][index1]++;
                        local_newClusterSize[tid][index2]++;
                        local_newClusterSize[tid][index3]++;
                        local_newClusterSize[tid][index4]++;
                        
                        local_newClusterSize[tid][index5]++;
                        local_newClusterSize[tid][index6]++;
                        local_newClusterSize[tid][index7]++;
                        local_newClusterSize[tid][index8]++;
                        
                        //n*d
                        local_newClusters[tid][index1][0] += objects[i][0];
                        local_newClusters[tid][index2][0] += objects[i+1][0];
                        local_newClusters[tid][index3][0] += objects[i+2][0];
                        local_newClusters[tid][index4][0] += objects[i+3][0];
                        local_newClusters[tid][index5][0] += objects[i+4][0];
                        local_newClusters[tid][index6][0] += objects[i+5][0];
                        local_newClusters[tid][index7][0] += objects[i+6][0];
                        local_newClusters[tid][index8][0] += objects[i+7][0];

                        local_newClusters[tid][index1][1] += objects[i][1];
                        local_newClusters[tid][index2][1] += objects[i+1][1];
                        local_newClusters[tid][index3][1] += objects[i+2][1];
                        local_newClusters[tid][index4][1] += objects[i+3][1];
                        local_newClusters[tid][index5][1] += objects[i+4][1];
                        local_newClusters[tid][index6][1] += objects[i+5][1];
                        local_newClusters[tid][index7][1] += objects[i+6][1];
                        local_newClusters[tid][index8][1] += objects[i+7][1];

                        local_newClusters[tid][index1][2] += objects[i][2];
                        local_newClusters[tid][index2][2] += objects[i+1][2];
                        local_newClusters[tid][index3][2] += objects[i+2][2];
                        local_newClusters[tid][index4][2] += objects[i+3][2];
                        local_newClusters[tid][index5][2] += objects[i+4][2];
                        local_newClusters[tid][index6][2] += objects[i+5][2];
                        local_newClusters[tid][index7][2] += objects[i+6][2];
                        local_newClusters[tid][index8][2] += objects[i+7][2];

                        local_newClusters[tid][index1][3] += objects[i][3];
                        local_newClusters[tid][index2][3] += objects[i+1][3];
                        local_newClusters[tid][index3][3] += objects[i+2][3];
                        local_newClusters[tid][index4][3] += objects[i+3][3];
                        local_newClusters[tid][index5][3] += objects[i+4][3];
                        local_newClusters[tid][index6][3] += objects[i+5][3];
                        local_newClusters[tid][index7][3] += objects[i+6][3];
                        local_newClusters[tid][index8][3] += objects[i+7][3];

                        local_newClusters[tid][index1][4] += objects[i][4];
                        local_newClusters[tid][index2][4] += objects[i+1][4];
                        local_newClusters[tid][index3][4] += objects[i+2][4];
                        local_newClusters[tid][index4][4] += objects[i+3][4];
                        local_newClusters[tid][index5][4] += objects[i+4][4];
                        local_newClusters[tid][index6][4] += objects[i+5][4];
                        local_newClusters[tid][index7][4] += objects[i+6][4];
                        local_newClusters[tid][index8][4] += objects[i+7][4];

                        local_newClusters[tid][index1][5] += objects[i][5];
                        local_newClusters[tid][index2][5] += objects[i+1][5];
                        local_newClusters[tid][index3][5] += objects[i+2][5];
                        local_newClusters[tid][index4][5] += objects[i+3][5];
                        local_newClusters[tid][index5][5] += objects[i+4][5];
                        local_newClusters[tid][index6][5] += objects[i+5][5];
                        local_newClusters[tid][index7][5] += objects[i+6][5];
                        local_newClusters[tid][index8][5] += objects[i+7][5];

                        local_newClusters[tid][index1][6] += objects[i][6];
                        local_newClusters[tid][index2][6] += objects[i+1][6];
                        local_newClusters[tid][index3][6] += objects[i+2][6];
                        local_newClusters[tid][index4][6] += objects[i+3][6];
                        local_newClusters[tid][index5][6] += objects[i+4][6];
                        local_newClusters[tid][index6][6] += objects[i+5][6];
                        local_newClusters[tid][index7][6] += objects[i+6][6];
                        local_newClusters[tid][index8][6] += objects[i+7][6];
                        
                        local_newClusters[tid][index1][7] += objects[i][7];
                        local_newClusters[tid][index2][7] += objects[i+1][7];
                        local_newClusters[tid][index3][7] += objects[i+2][7];
                        local_newClusters[tid][index4][7] += objects[i+3][7];
                        local_newClusters[tid][index5][7] += objects[i+4][7];
                        local_newClusters[tid][index6][7] += objects[i+5][7];
                        local_newClusters[tid][index7][7] += objects[i+6][7];
                        local_newClusters[tid][index8][7] += objects[i+7][7];
                    }
                } /* end of #pragma omp parallel */
            
            } else {
            
            
            
#pragma omp parallel \
shared(objects,clusters,membership,local_newClusters,local_newClusterSize)
            {
                int tid = omp_get_thread_num();
#pragma omp for \
private(i,j) \
firstprivate(numObjs,numClusters,numCoords) \
schedule(static) \
reduction(+:delta2)
                //firstprivate: Listed variables are initialized according to the value of their original objects prior to entry into the parallel or work-sharing construct.
                for (i=0; i<ilim; i+=4) {
                    /* find the array index of nestest cluster center */
                    //n*k*d
                    int index1 = find_nearest_cluster(numClusters, numCoords,
                                                      objects[i], clusters);
                    
                    int index2 = find_nearest_cluster(numClusters, numCoords,
                                                      objects[i+1], clusters);
                    int index3 = find_nearest_cluster(numClusters, numCoords,
                                                      objects[i+2], clusters);
                    
                    int index4 = find_nearest_cluster(numClusters, numCoords,
                                                      objects[i+3], clusters);
                    
                    if (membership[i] != index1) delta2 ++;
                    if (membership[i+1] != index2) delta2 ++;
                    if (membership[i+2] != index3) delta2 ++;
                    if (membership[i+3] != index4) delta2 ++;
                    
                    /* assign the membership to object i */
                    membership[i] = index1;
                    membership[i+1] = index2;
                    membership[i+2] = index3;
                    membership[i+3] = index4;
                    
                    /* update new cluster centers : sum of all objects located
                     within (average will be performed later) */
                    local_newClusterSize[tid][index1]++;
                    local_newClusterSize[tid][index2]++;
                    local_newClusterSize[tid][index3]++;
                    local_newClusterSize[tid][index4]++;
                    
                    //n*d
                    for (j=0; j<numCoords; j++){
                        local_newClusters[tid][index1][j] += objects[i][j];
                        local_newClusters[tid][index2][j] += objects[i+1][j];
                        local_newClusters[tid][index3][j] += objects[i+2][j];
                        local_newClusters[tid][index4][j] += objects[i+3][j];
                    }
                }
            } /* end of #pragma omp parallel */
                
                
            }
            for (i = ilim; i<numObjs; i++) {
                /* find the array index of nestest cluster center */
                index = find_nearest_cluster(numClusters, numCoords,
                                             objects[i], clusters);
                
                /* if membership changes, increase delta by 1 */
                if (membership[i] != index) delta2 ++;
                
                /* assign the membership to object i */
                membership[i] = index;
                
                /* update new cluster centers : sum of all objects located
                 within (average will be performed later) */
                local_newClusterSize[0][index]++;
                for (j=0; j<numCoords; j++)
                    local_newClusters[0][index][j] += objects[i][j];
            }
            /* let the main thread perform the array reduction */
            for (i=0; i<numClusters; i++) {
                for (j=0; j<nthreads; j++) {
                    newClusterSize[i] += local_newClusterSize[j][i];
                    local_newClusterSize[j][i] = 0.0;
                    for (k=0; k<numCoords; k++) {
                        newClusters[i][k] += local_newClusters[j][i][k];
                        local_newClusters[j][i][k] = 0.0;
                    }
                }
            }
        }
        
        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 1)
                    clusters[i][j] = newClusters[i][j] / newClusterSize[i];
                newClusters[i][j] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }
        
        //        delta /= numObjs;
    } while (delta2 > threshold*numObjs && loop++ < 500);
    
    if (_debug) {
        timing = omp_get_wtime() - timing;
        printf("nloops = %2d (T = %7.4f)",loop,timing);
    }
    
    if (!is_perform_atomic) {
        free(local_newClusterSize[0]);
        free(local_newClusterSize);
        
        for (i=0; i<nthreads; i++)
            for (j=0; j<numClusters; j++)
                free(local_newClusters[i][j]);
        free(local_newClusters[0]);
        free(local_newClusters);
    }
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);
    
    return clusters;
}


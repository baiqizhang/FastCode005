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


/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__inline static
float euclid_dist_2(int    numdims,  /* no. dimensions */
                    float *coord1,   /* [numdims] */
                    float *coord2)   /* [numdims] */
{
    int i;
    float ans=0.0;
    
    for (i=0; i<numdims-4; i+=4){
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);
        ans += (coord1[i+1]-coord2[i+1]) * (coord1[i+1]-coord2[i+1]);
        ans += (coord1[i+2]-coord2[i+2]) * (coord1[i+2]-coord2[i+2]);
        ans += (coord1[i+3]-coord2[i+3]) * (coord1[i+3]-coord2[i+3]);
    }
    for (; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);
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
    
//    float ans1=0.0,ans2=0.0,ans3=0.0;
//    int i;
//
//    for (i=0; i<numCoords-4; i+=4){
//        ans1 += (object[i] - clusters[0][i])  *  (object[i]-  clusters[0][i]);
//        ans1 += (object[i+1]-clusters[0][i+1]) * (object[i+1]-clusters[0][i+1]);
//        ans1 += (object[i+2]-clusters[0][i+2]) * (object[i+2]-clusters[0][i+2]);
//        ans1 += (object[i+3]-clusters[0][i+3]) * (object[i+3]-clusters[0][i+3]);
//
//        ans2 += (object[i] - clusters[1][i])  *  (object[i]-  clusters[1][i]);
//        ans2 += (object[i+1]-clusters[1][i+1]) * (object[i+1]-clusters[1][i+1]);
//        ans2 += (object[i+2]-clusters[1][i+2]) * (object[i+2]-clusters[1][i+2]);
//        ans2 += (object[i+3]-clusters[1][i+3]) * (object[i+3]-clusters[1][i+3]);
//
//        ans3 += (object[i] - clusters[2][i])  *  (object[i]-  clusters[2][i]);
//        ans3 += (object[i+1]-clusters[2][i+1]) * (object[i+1]-clusters[2][i+1]);
//        ans3 += (object[i+2]-clusters[2][i+2]) * (object[i+2]-clusters[2][i+2]);
//        ans3 += (object[i+3]-clusters[2][i+3]) * (object[i+3]-clusters[2][i+3]);
//
//    }
//    for (; i<numCoords; i++){
//        ans1 += (object[i] - clusters[0][i])  *  (object[i]-  clusters[0][i]);
//        ans2 += (object[i] - clusters[1][i])  *  (object[i]-  clusters[1][i]);
//        ans3 += (object[i] - clusters[2][i])  *  (object[i]-  clusters[2][i]);
//    }
//
//    if (ans1<ans2&&ans1<ans3)
//        return 0;
//    if (ans2<ans1&&ans2<ans3)
//        return 1;
//    return(2);
    return index;
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
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                              new cluster */
    float    delta;          /* % of objects change their clusters */
    float  **clusters;       /* out: [numClusters][numCoords] */
    float  **newClusters;    /* [numClusters][numCoords] */
    double   timing;
    
    int      nthreads;             /* no. threads */
    int    **local_newClusterSize; /* [nthreads][numClusters] */
    float ***local_newClusters;    /* [nthreads][numClusters][numCoords] */
    
    int mask = 4;
    if (numObjs>10000)
        mask = 7;
    int ilim = numObjs-(numObjs&mask);

    
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
    
    /* pick first numClusters elements of objects[] as initial cluster centers*/
#pragma omp parallel for private(i,j) schedule(static) collapse(2) // added by Vincent
    for (i=0; i<numClusters; i++)
        for (j=0; j<numCoords; j++)
            clusters[i][j] = objects[i][j];
    
    /* initialize membership[] */
    // loop unrolling - added by Vincent
#pragma omp parallel for
    for (i=0; i<numObjs-3; i += 4) {
        membership[i] = -1;
        membership[i+1] = -1;
        membership[i+2] = -1;
        membership[i+3] = -1;
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
    
    if (_debug) timing = omp_get_wtime();
    do {
        delta = 0.0;

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
#pragma omp parallel \
shared(objects,clusters,membership,local_newClusters,local_newClusterSize)
            {
                int tid = omp_get_thread_num();
#pragma omp for \
private(i,j) \
firstprivate(numObjs,numClusters,numCoords) \
schedule(static) \
reduction(+:delta)
                //firstprivate: Listed variables are initialized according to the value of their original objects prior to entry into the parallel or work-sharing construct.
                int index1,index2,index3,index4,index5,index6,index7,index8;
                for (i=0; i<ilim; i+=(mask+1)) {
                    /* find the array index of nestest cluster center */
                    int index1 = find_nearest_cluster(numClusters, numCoords,
                                                 objects[i], clusters);
                    
                    int index2 = find_nearest_cluster(numClusters, numCoords,
                                                      objects[i+1], clusters);
                    int index3 = find_nearest_cluster(numClusters, numCoords,
                                                      objects[i+2], clusters);
                    
                    int index4 = find_nearest_cluster(numClusters, numCoords,
                                                      objects[i+3], clusters);

                    if (maks == 7){
                        index5 = find_nearest_cluster(numClusters, numCoords,
                                                          objects[i+4], clusters);
                        index6 = find_nearest_cluster(numClusters, numCoords,
                                                          objects[i+5], clusters);
                        index7 = find_nearest_cluster(numClusters, numCoords,
                                                          objects[i+6], clusters);
                        
                        index8 = find_nearest_cluster(numClusters, numCoords,
                                                          objects[i+7], clusters);
                    }

                    /* if membership changes, increase delta by 1 */
                    if (membership[i] != index1) delta += 1.0;
                    if (membership[i+1] != index2) delta += 1.0;
                    if (membership[i+2] != index3) delta += 1.0;
                    if (membership[i+3] != index4) delta += 1.0;

                    if (maks == 7){
                        if (membership[i+4] != index5) delta += 1.0;
                        if (membership[i+5] != index6) delta += 1.0;
                        if (membership[i+6] != index7) delta += 1.0;
                        if (membership[i+7] != index8) delta += 1.0;
                    }
                    
                    /* assign the membership to object i */
                    membership[i] = index1;
                    membership[i+1] = index2;
                    membership[i+2] = index3;
                    membership[i+3] = index4;

                    if (maks == 7){
                        membership[i+4] = index5;
                        membership[i+5] = index6;
                        membership[i+6] = index7;
                        membership[i+7] = index8;
                    }
                    
                    /* update new cluster centers : sum of all objects located
                     within (average will be performed later) */
                    local_newClusterSize[tid][index1]++;
                    local_newClusterSize[tid][index2]++;
                    local_newClusterSize[tid][index3]++;
                    local_newClusterSize[tid][index4]++;

                    if (maks == 7){
                        local_newClusterSize[tid][index5]++;
                        local_newClusterSize[tid][index6]++;
                        local_newClusterSize[tid][index7]++;
                        local_newClusterSize[tid][index8]++;
                    }
                    
                    for (j=0; j<numCoords; j++){
                        local_newClusters[tid][index1][j] += objects[i][j];
                        local_newClusters[tid][index2][j] += objects[i+1][j];
                        local_newClusters[tid][index3][j] += objects[i+2][j];
                        local_newClusters[tid][index4][j] += objects[i+3][j];

                        if (maks == 7){
                            local_newClusters[tid][index5][j] += objects[i+4][j];
                            local_newClusters[tid][index6][j] += objects[i+1][j];
                            local_newClusters[tid][index7][j] += objects[i+2][j];
                            local_newClusters[tid][index8][j] += objects[i+3][j];
                        }
                    }
                }
            } /* end of #pragma omp parallel */
            for (i = ilim; i<numObjs; i++) {
                /* find the array index of nestest cluster center */
                index = find_nearest_cluster(numClusters, numCoords,
                                             objects[i], clusters);
                
                /* if membership changes, increase delta by 1 */
                if (membership[i] != index) delta += 1.0;
                
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
        
        delta /= numObjs;
    } while (delta > threshold && loop++ < 500);
    
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


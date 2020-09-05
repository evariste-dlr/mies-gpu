#ifndef DMIES_H
#define DMIES_H

#include "sparse.h"

/* Large float value (2^32) */
#ifndef INFTY
 #define INFTY 429496729.0
#endif


extern float GPU_ELAPSED_TIME;

/*
__global__
void d_mies_survivor_round( const sparse_mat * adj,
                            sparse_mat * surv,
                            uint nnz,
                            const uint * n_candidates );

__global__
void d_mies_candidate_round( sparse_mat * adj,
                             const sparse_mat * surv,
                             uint nnz,
                             uint * n_candidates );
*/


void d_mies( sparse_mat & adj, sparse_mat & surv );



#endif

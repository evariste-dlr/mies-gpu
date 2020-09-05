#ifndef HMIES_H
#define HMIES_H

#include "sparse.h"

/* Large float value (2^32) */
#ifndef INFTY
 //#define INF 429496729.0
 #define INFTY 1000.0
#endif


/**
 * @brief Update the survivors wrt each node's neighborhood (CPU version)
 *
 * @param adj   Adjacency matrix
 * @param surv  Survivors
 * @param mins  Minimum value of each neighboring candidate edge
 * @param n_candidates   Number of candidate edges
 */
void h_mies_survivor_round( const sparse_mat & adj,
                            sparse_mat & surv,
                            float* mins,
                            const uint & n_candidates );


/**
 * @brief Update the candidates wrt each node's neighborhood (CPU version)
 *
 * @param adj   Adjacency matrix
 * @param surv  Survivors
 * @param exist_surv     For each row, true iff there is a survivor in this row
 * @param n_candidates   Number of candidate edges
 */
void h_mies_candidate_round( sparse_mat & adj,
                             const sparse_mat & surv,
                             bool* exist_surv,
                             uint & n_candidates );


/**
 * @brief Find a Maximal Independant Edge Set in a graph (CPU version)
 *
 */
void h_mies( sparse_mat & adj, sparse_mat & surv,
             float* mins, bool* exist_surv );


#endif

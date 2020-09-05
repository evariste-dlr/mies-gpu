#include <iostream>
#include "h_mies.h"


void h_mies_survivor_round( const sparse_mat & adj,
			    sparse_mat & surv,
			    float * mins,
			    const uint & n_candidates )
{

  // Find the minimum for each line
  for (int i=0; i<adj.rows; i++){
    float min = INFTY;
    for (int j=adj.csr_row_ptr[i]; j<adj.csr_row_ptr[i+1]; j++)
      if (min > adj.val[j])
	min = adj.val[j];

    mins[i] = min;
  }

  for (int e=0; e<adj.nnz; e++){

    // If e is candidate
    if (adj.val[e] != INFTY){

      int i = adj.coo_row_ind[e];
      int j = adj.coo_col_ind[e];

      float local_min = (mins[i] < mins[j]) ? mins[i] : mins[j];

      // Edge e is a survivor if: (e is a survivor) OR (e is a local min)
      if (local_min == adj.val[e])
	surv.val[e] = 1.0;
    }
  }
}


void h_mies_candidate_round( sparse_mat & adj,
			     const sparse_mat & surv,
			     bool* exist_surv,
			     uint & n_candidates )
{

  for (int i=0; i<adj.rows; i++){
    bool exist = exist_surv[i];

    for (int j=adj.csr_row_ptr[i]; j<adj.csr_row_ptr[i+1] && !exist; j++)
      exist = surv.val[j] > 0.0;

    exist_surv[i] = exist;
  }

  for (int e=0; e<adj.nnz; e++){

    // If e is candidate
    if (adj.val[e] != INFTY){

      int i = adj.coo_row_ind[e];
      int j = adj.coo_col_ind[e];

      //if (surv_neighbors){
      if (exist_surv[i] || exist_surv[j]){
	adj.val[e] = INFTY;
	n_candidates--;
      }
    }
  }
}


void h_mies( sparse_mat & adj, sparse_mat & surv,
	     float* mins, bool* exist_surv )
{
  uint n_candidates = adj.nnz;

  bool clean_up = false;
  if (!mins || !exist_surv){
    mins = (float*) malloc(adj.rows * sizeof(float));
    exist_surv = (bool*) calloc(adj.rows, sizeof(bool));
    clean_up = true;
  }

  while (n_candidates > 0){
    h_mies_survivor_round( adj, surv, mins, n_candidates );
    h_mies_candidate_round( adj, surv, exist_surv, n_candidates );
  }

  if (clean_up){
    free(mins);
    free(exist_surv);
  }
}


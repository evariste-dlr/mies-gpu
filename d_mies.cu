#include <iostream>
#include "d_mies.h"

#define TH_NODES_PER_BLOCK 128
#define TH_EDGES_PER_BLOCK 128
#define EDGES_PER_THREAD 1
#define NODES_PER_THREAD 1

float GPU_ELAPSED_TIME=-1.0;


void cudaTimerStart(cudaEvent_t &start) {
        cudaEventCreate(&start);
        cudaEventRecord(start, 0 );
}

float cudaTimerStop(cudaEvent_t &start) {
        cudaEvent_t stop;
        cudaEventCreate(&stop);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float time;
        cudaEventElapsedTime(&time, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return time;
}


__global__
void d_mies_min_row( const sparse_mat* adj,
                     float* mins,
                     uint rows )
{
  uint n = blockDim.x * blockIdx.x + threadIdx.x;

  if (n < rows){

    float min = INFTY;
    for (int i=adj->csr_row_ptr[n]; i<adj->csr_row_ptr[n+1]; i++)
      if (min > adj->val[i])
        min = adj->val[i];

    mins[n] = min;
  }
}


__global__
void d_mies_survivor_round( const sparse_mat * adj,
                            sparse_mat * surv,
                            const float* mins,
                            uint nnz,
                            const uint * n_candidates )
{
  uint e = blockDim.x * blockIdx.x + threadIdx.x;

  // If e is candidate
  if (e < nnz){
   if(adj->val[e] != INFTY){

    int i = adj->coo_row_ind[e];
    int j = adj->coo_col_ind[e];

    float local_min = (mins[i] < mins[j]) ? mins[i] : mins[j];

    // Edge e is a survivor if: (e is a survivor) OR (e is a local min)
    if (local_min == adj->val[e])
      surv->val[e] = 1.0;
   }
  }
}


__global__
void d_mies_surv_row( const sparse_mat* surv,
                      bool* exist_surv,
                      uint rows )
{
  uint n = blockDim.x * blockIdx.x + threadIdx.x;

  if (n < rows){

    bool exist = exist_surv[n];
    for ( int i=surv->csr_row_ptr[n];
          i<surv->csr_row_ptr[n+1] && !exist;
          i++ )
      exist = surv->val[i] > 0.0;

    exist_surv[n] = exist;
  }
}


__global__
void d_mies_candidate_round( sparse_mat * adj,
                             const sparse_mat * surv,
                             const bool* exist_surv,
                             uint nnz,
                             uint * n_candidates )
{
  uint e = blockDim.x * blockIdx.x + threadIdx.x;

  // If e is candidate
  if (e < nnz){
   if (adj->val[e] != INFTY){

    int i = adj->coo_row_ind[e];
    int j = adj->coo_col_ind[e];

    if (exist_surv[i] || exist_surv[j]){
      adj->val[e] = INFTY;
      atomicSub(n_candidates, 1);
    }
   }
  }
}



void d_mies( sparse_mat & adj, sparse_mat & surv )
{
  cudaSetDevice(0);

  cudaEvent_t start;
  //cudaTimerStart(start);


  /* ======== Upload the matrices onto the GPU ======== */
  
  uint n_candidates = adj.nnz;
  uint* d_n_candidates;
  cudaMalloc( &d_n_candidates, sizeof(uint) );
  cudaMemcpy( d_n_candidates, &n_candidates, sizeof(uint), cudaMemcpyHostToDevice );

  float* d_adj_val;
  uint* d_adj_coo_row_ind;
  uint* d_adj_coo_col_ind;
  uint* d_adj_csr_row_ptr;
  float* d_surv_val;
  uint* d_surv_coo_row_ind;
  uint* d_surv_coo_col_ind;
  uint* d_surv_csr_row_ptr;

  cudaMalloc(&d_adj_val, adj.nnz * sizeof(float));
  cudaMalloc(&d_adj_coo_row_ind, adj.nnz * sizeof(uint));
  cudaMalloc(&d_adj_coo_col_ind, adj.nnz * sizeof(uint));
  cudaMalloc(&d_adj_csr_row_ptr, (adj.rows+1) * sizeof(uint));

  cudaMalloc(&d_surv_val, surv.nnz * sizeof(float));
  cudaMalloc(&d_surv_coo_row_ind, surv.nnz * sizeof(uint));
  cudaMalloc(&d_surv_coo_col_ind, surv.nnz * sizeof(uint));
  cudaMalloc(&d_surv_csr_row_ptr, (surv.rows+1) * sizeof(uint));

  float* d_mins;
  bool* d_exist_surv;
  cudaMalloc(&d_mins, adj.rows * sizeof(float));
  cudaMemset( d_mins, 0.0, adj.rows * sizeof(float));
  cudaMalloc(&d_exist_surv, adj.rows * sizeof(bool));
  cudaMemset( d_exist_surv, 0, adj.rows * sizeof(bool));

  sparse_mat * d_adj;
  sparse_mat * d_surv;
  cudaMalloc(&d_adj, sizeof(sparse_mat));
  cudaMalloc(&d_surv, sizeof(sparse_mat));

  // Bind the addresses
  sparse_mat h_adj_bind = {
    .val = d_adj_val,
    .coo_row_ind = d_adj_coo_row_ind,
    .coo_col_ind = d_adj_coo_col_ind,
    .csr_row_ptr = d_adj_csr_row_ptr,
    .nnz = adj.nnz,
    .rows = adj.rows,
    .cols = adj.cols
  };

  sparse_mat h_surv_bind = {
    .val = d_surv_val,
    .coo_row_ind = d_surv_coo_row_ind,
    .coo_col_ind = d_surv_coo_col_ind,
    .csr_row_ptr = d_surv_csr_row_ptr,
    .nnz = surv.nnz,
    .rows = surv.rows,
    .cols = surv.cols
  };

  cudaMemcpy( d_adj, &h_adj_bind, sizeof(sparse_mat), cudaMemcpyHostToDevice );
  cudaMemcpy( d_surv, &h_surv_bind, sizeof(sparse_mat), cudaMemcpyHostToDevice );

  cudaMemcpy( d_adj_val, adj.val, adj.nnz*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_adj_coo_row_ind, adj.coo_row_ind, adj.nnz*sizeof(uint), cudaMemcpyHostToDevice );
  cudaMemcpy( d_adj_coo_col_ind, adj.coo_col_ind, adj.nnz*sizeof(uint), cudaMemcpyHostToDevice );
  cudaMemcpy( d_adj_csr_row_ptr, adj.csr_row_ptr, (adj.rows+1)*sizeof(uint), cudaMemcpyHostToDevice );

  cudaMemcpy( d_surv_val, surv.val, surv.nnz*sizeof(float), cudaMemcpyHostToDevice );
  cudaMemcpy( d_surv_coo_row_ind, surv.coo_row_ind, surv.nnz*sizeof(uint), cudaMemcpyHostToDevice );
  cudaMemcpy( d_surv_coo_col_ind, surv.coo_col_ind, surv.nnz*sizeof(uint), cudaMemcpyHostToDevice );
  cudaMemcpy( d_surv_csr_row_ptr, surv.csr_row_ptr, (surv.rows+1)*sizeof(uint), cudaMemcpyHostToDevice );


  /* ======== Main Loop  ======== */

  dim3 dimBlockNodes(TH_NODES_PER_BLOCK, 1, 1);
  dim3 dimGridNodes(ceil((double)(adj.rows) / (double)TH_NODES_PER_BLOCK*NODES_PER_THREAD), 1, 1);
  dim3 dimBlockEdges(TH_EDGES_PER_BLOCK, 1, 1);
  dim3 dimGridEdges(ceil((double)(adj.nnz) / ((double)TH_EDGES_PER_BLOCK)*EDGES_PER_THREAD), 1, 1);

  cudaTimerStart(start);

  bool* h_exist_surv = (bool*) calloc(adj.rows, sizeof(bool));
  bool* h_eq_surv = (bool*) calloc(adj.rows, sizeof(bool));

  while (n_candidates > 0){
    d_mies_min_row<<<dimGridNodes, dimBlockNodes>>>( d_adj, d_mins, adj.rows );
    d_mies_survivor_round<<<dimGridEdges, dimBlockEdges>>>( d_adj, d_surv, d_mins, adj.nnz, d_n_candidates );

    d_mies_surv_row<<<dimGridNodes, dimBlockNodes>>>( d_surv, d_exist_surv, surv.rows );
    d_mies_candidate_round<<<dimGridEdges, dimBlockEdges>>>( d_adj, d_surv, d_exist_surv, adj.nnz, d_n_candidates );

    cudaMemcpy( &n_candidates, d_n_candidates, sizeof(uint), cudaMemcpyDeviceToHost );
  }

  cudaMemcpy( surv.val, d_surv_val, surv.nnz*sizeof(float), cudaMemcpyDeviceToHost );

  GPU_ELAPSED_TIME = cudaTimerStop(start);


  /* ======== Clean up  ======== */

  cudaFree(d_adj);
  cudaFree(d_surv);

  cudaFree(d_mins);
  cudaFree(d_exist_surv);

  cudaFree(d_adj_val);
  cudaFree(d_adj_coo_row_ind);
  cudaFree(d_adj_coo_col_ind);
  cudaFree(d_adj_csr_row_ptr);

  cudaFree(d_surv_val);
  cudaFree(d_surv_coo_row_ind);
  cudaFree(d_surv_coo_col_ind);
  cudaFree(d_surv_csr_row_ptr);

  cudaFree(d_n_candidates);
}

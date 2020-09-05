#include <iostream>
#include <list>
#include <algorithm>
#include <random>

#include <cstring>

#include "sparse.h"

#define NV 2000000   // Total number of nodes
#define MEAN_SIZE 100// Mean size of the graphs
#define MIN_SIZE 20  // Minimium graph size (check wrt density)
#define MAX_SIZE 10  // Maximum size of each graph
#define STD_SIZE 8   // Standard deviation of the graphs' sizes
#define DENSITY 0.7  // Density of the graphs

#define MEAN_WEIGHT 20
#define STD_WEIGHT 4


const std::string RESET         = "\x1b[0m";
const std::string BLINK         = "\x1b[5m";
const std::string GREEN         = "\x1b[92m";
const std::string GREEN_BACK    = "\x1b[42m\x1b[1m";
const std::string RED           = "\x1b[91m\x1b[1m";
const std::string RED_BACK      = "\x1b[101m";
const std::string CYAN          = "\x1b[96m";



void generate_graphs( sparse_mat** adj, sparse_mat** surv )
{
  int n_vertices = 0;
  std::list<float*> graphs;
  std::list<size_t> sizes;

  std::default_random_engine gen;
  std::normal_distribution<float> dist_size(MEAN_SIZE, STD_SIZE);
  std::normal_distribution<float> dist_val(MEAN_WEIGHT, STD_WEIGHT);

  uint size;

  do{
    float fsize = dist_size(gen);
    if (fsize < 0.0) size = 0;
    else             size = (uint) fsize;

    if (size < MIN_SIZE)
      size = MIN_SIZE;

    if (size > MAX_SIZE)
      size = MAX_SIZE;

    if (n_vertices + size >= NV)
      size = NV - n_vertices;

    /* ======= CREATE A GRAPH ======= */
    
    // max number of edges
    uint max_edges = (size*(size-1))/2;
    
    float* g = (float*) malloc(size * size * sizeof(float));
    std::vector<float> values(max_edges, 0.0);

    // number of edges
    uint n_edges = (uint)(DENSITY * max_edges);
    
    for (uint i=0; i<n_edges; i++)
      values[i] = dist_val(gen);

    std::random_shuffle(values.begin(), values.end());

    // fill the matrix
    uint row=0, col=1;
    for (uint i=0; i<max_edges; i++){
      g[IND(row, col, size)] = values[i];
      g[IND(col, row, size)] = values[i];
      col++;
      if (col >= size){
        row++; col=row+1;
      }
    }

    /* ======= STORE THE GRAPH ======= */

    graphs.push_back(g);
    sizes.push_back(size);

    n_vertices += size;

  } while(n_vertices < NV);

  *adj = to_sparse_bd(graphs, sizes);

  
  /* ======= CREATE SURV MATRIX ======= */

  // Surv's structure is the same of adj but it's empty
  *surv = (sparse_mat*) malloc(sizeof(sparse_mat));
  (*surv)->val = (float*) calloc((*adj)->nnz, sizeof(float));
  (*surv)->coo_row_ind = (uint*) calloc((*adj)->nnz, sizeof(uint));
  (*surv)->coo_col_ind = (uint*) calloc((*adj)->nnz, sizeof(uint));
  (*surv)->csr_row_ptr = (uint*) calloc((*adj)->rows+1, sizeof(uint));
  (*surv)->nnz = (*adj)->nnz;
  (*surv)->rows = (*adj)->rows;
  (*surv)->cols = (*adj)->cols;

  memcpy((*surv)->coo_row_ind, (*adj)->coo_row_ind, sizeof(uint)*(*adj)->nnz);
  memcpy((*surv)->coo_col_ind, (*adj)->coo_col_ind, sizeof(uint)*(*adj)->nnz);
  memcpy((*surv)->csr_row_ptr, (*adj)->csr_row_ptr, sizeof(uint)*((*adj)->rows+1));


  /* ======= CLEAN UP ====== */

  for ( std::list<float*>::iterator it = graphs.begin();
        it != graphs.end(); it++ )
    free(*it);

}


/**
 * @brief Check if the surviving edges constitute a Maximal Independant Set
 */
void verify_mies(const sparse_mat & adj, const sparse_mat & surv)
{
  /* ====== Check independence ===== */
  bool inde = true;
  for (int e=0; e<adj.nnz && inde; e++){

    if (surv.val[e] > 0.0){
      uint i = adj.coo_row_ind[e];
      uint j = adj.coo_col_ind[e];

      // Check if the neighbors are not surviving
      for (int k=adj.csr_row_ptr[i]; k<adj.csr_row_ptr[i+1] && inde; k++)
        if (surv.coo_col_ind[k] != j)
          inde = surv.val[k] <= 0.0;

      for (int l=adj.csr_row_ptr[j]; l<adj.csr_row_ptr[j+1] && inde; l++)
        if (surv.coo_col_ind[l] != i)
        inde = surv.val[l] <= 0.0;
    }
  }

  /* ====== Check Maximality ===== */
  bool maxim = true;
  for (int e=0; e<adj.nnz && maxim; e++){

    bool linked_to_surv = false;

    if (surv.val[e] <= 0.0){
      uint i = adj.coo_row_ind[e];
      uint j = adj.coo_col_ind[e];

      // Check that there is a surviving neighbor
      for (int k=adj.csr_row_ptr[i]; k<adj.csr_row_ptr[i+1] && !linked_to_surv; k++)
        if (surv.coo_col_ind[k] != j)
          linked_to_surv = surv.val[k] > 0.0;

      for (int l=adj.csr_row_ptr[j]; l<adj.csr_row_ptr[j+1] && !linked_to_surv; l++)
        if (surv.coo_col_ind[l] != i)
          linked_to_surv = surv.val[l] > 0.0;

      maxim = linked_to_surv;
    }
  }

  if (inde && maxim)
    std::cout << "    " << GREEN << "[MIES]" << RESET << std::endl;

  else{
    std::cout << "    " << RED << "[NOT MIES]" << RESET;
    if (!inde)
      std::cout << "           " << "Not independent set" << std::endl;
    else
      std::cout << "           " << "Not maximal" << std::endl;
  }
}

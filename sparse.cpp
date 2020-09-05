#include "sparse.h"

#include <iostream>
#include <list>

#include <cstring>

sparse_mat * to_sparse_bd( const std::list<float*> & values,
                           const std::list<size_t> & rows)
{
  if (values.size() != rows.size()){
    std::cerr << "[E] Different number of graphs" << std::endl;
    return NULL;
  }
  
  // Get total nnz values
  uint nnz = 0;
  std::list<float*>::const_iterator it_v = values.begin();
  for (std::list<size_t>::const_iterator it_n=rows.begin(); it_n!=rows.end(); it_n++, it_v++)
    for (uint i=0; i<(*it_n); i++)
      for (uint j=0; j<(*it_n); j++)
        if ((*it_v)[IND(i,j,(*it_n))] != 0.0)
          nnz++;

  // Get total number of nodes
  uint n_nodes = 0;
  for (std::list<size_t>::const_iterator it_n=rows.begin(); it_n!=rows.end(); it_n++)
    n_nodes += (*it_n);

  sparse_mat* mat = (sparse_mat*) malloc(sizeof(sparse_mat));
  mat->val = (float*) malloc(nnz * sizeof(float));
  mat->coo_row_ind = (uint*) malloc(nnz * sizeof(uint));
  mat->coo_col_ind = (uint*) malloc(nnz * sizeof(uint));
  mat->csr_row_ptr = (uint*) malloc((n_nodes+1) * sizeof(uint));
  mat->nnz = nnz;
  mat->rows = n_nodes;
  mat->cols = n_nodes;

  // Fill in the matrix
  uint first_node = 0;
  uint idx = 0;
  bool newline = false;

  it_v = values.begin();
  for (std::list<size_t>::const_iterator it_n=rows.begin(); it_n!=rows.end(); it_n++, it_v++){

    for (uint i=0; i<(*it_n); i++){
      // if the previous line is empty (no newline=false bellow)
      if (newline)
        mat->csr_row_ptr[first_node+i-1] = idx;

      newline = true;
      for (uint j=0; j<(*it_n); j++){

        if ((*it_v)[IND(i,j,(*it_n))] != 0.0){
          mat->val[idx] = (*it_v)[IND(i,j,(*it_n))];
          mat->coo_row_ind[idx] = first_node + i;
          mat->coo_col_ind[idx] = first_node + j;

          if (newline){
            mat->csr_row_ptr[first_node+i] = idx;
            newline = false;
          }

          idx++;
        }

      } // for j
    } // for i

    first_node += (*it_n);
  }// for n

  // If the last line is empty
  if (newline)
    mat->csr_row_ptr[n_nodes-1] = nnz-1;

  // Last row ptr is the last data index
  mat->csr_row_ptr[n_nodes] = nnz-1;

  return mat;
}



void print_sparse( const sparse_mat & mat )
{
  for (uint i=0; i<mat.nnz; i++){
    if (mat.val[i] != 0.0){
      std::cout << "(" << mat.coo_row_ind[i] << "; "
                << mat.coo_col_ind[i] << ") \t -> "
                << mat.val[i] << std::endl;
    }
  }
  std::cout << std::endl;
}



void print_sparse_py( const sparse_mat & mat )
{
  std::cout << "data = [";
  for (int e=0; e<mat.nnz; e++)
    if (mat.val[e] != 0.0)
      std::cout << mat.val[e] << ", ";
  std::cout << "]" << std::endl;

  std::cout << "rows = [";
  for (int e=0; e<mat.nnz; e++)
    if (mat.val[e] != 0.0)
      std::cout << mat.coo_row_ind[e] << ", ";
  std::cout << "]" << std::endl;

  std::cout << "cols = [";
  for (int e=0; e<mat.nnz; e++)
    if (mat.val[e] != 0.0)
      std::cout << mat.coo_col_ind[e] << ", ";
  std::cout << "]" << std::endl;
}



void copy_sparse( sparse_mat * dst, const sparse_mat * src )
{
  dst->val = (float*) malloc(src->nnz * sizeof(float));
  dst->coo_row_ind = (uint*) malloc(src->nnz * sizeof(uint));
  dst->coo_col_ind = (uint*) malloc(src->nnz * sizeof(uint));
  dst->csr_row_ptr = (uint*) malloc((src->rows+1) * sizeof(uint));
  dst->nnz = src->nnz;
  dst->rows = src->rows;
  dst->cols = src->cols;

  memcpy( dst->val, src->val, dst->nnz * sizeof(float) );
  memcpy( dst->coo_row_ind, src->coo_row_ind, dst->nnz * sizeof(uint) );
  memcpy( dst->coo_col_ind, src->coo_col_ind, dst->nnz * sizeof(uint) );
  memcpy( dst->csr_row_ptr, src->csr_row_ptr, (dst->rows+1) * sizeof(uint) );
}


void delete_sparse( sparse_mat * mat )
{
  free(mat->val);
  free(mat->coo_row_ind);
  free(mat->coo_col_ind);
  free(mat->csr_row_ptr);
}

#ifndef SPARSE_H
#define SPARSE_H

#include <list>
#include <cstdlib>

#define IND(i,j,n)  (i*n+j)

typedef unsigned int uint;


/**
 * @brief Sparse matrix combining COO and CSR formats
 */
typedef struct {
  float* val;          //!< Values (array of size nnz)
  uint* coo_row_ind;   //!< Row COOrdinates of the values (size nnz)
  uint* coo_col_ind;   //!< Col COOrdinates of the values (size nnz)
  uint* csr_row_ptr;   //!< Indices of the first non-zero entry of each row (size rows+1)
  uint nnz;            //!< Number of non-zero entries
  uint rows;           //!< Number of rows
  uint cols;           //!< Number of columns
} sparse_mat;



/**
 * @brief Convert list of squared 2d arrays to a block-diagonal sparse matrix
 * @param values  Arrays (stored in row first format)
 * @param rows    Size of each array
 * @return A pointer to a heap-allocated (block diagonal) sparse matrix, the ith block corresponds to the ith array in the list
 */
sparse_mat * to_sparse_bd( const std::list<float*> & values,
                           const std::list<size_t> & rows);

/**
 * @prief Print the values stored in a sparse matrix
 */
void print_sparse( const sparse_mat & mat );

/**
 * @brief Print the matrix as a python's scipy csr_matrix
 */
void print_sparse_py( const sparse_mat & mat );


void copy_sparse( sparse_mat * dst, const sparse_mat * src );
void delete_sparse( sparse_mat * mat );

#endif

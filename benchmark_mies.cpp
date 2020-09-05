
#include <iostream>
#include <time.h>

#include "benchmark_utils.h"
#include "sparse.h"
#include "h_mies.h"
#include "d_mies.h"

#define NB_TESTS 10

double run_cpu( const sparse_mat & adj, const sparse_mat & surv )
{
  std::cout << " Compute MIES on CPU..." << std::endl;

  sparse_mat host_adj;
  sparse_mat host_surv;
  
  float* mins = (float*) malloc(adj.rows * sizeof(float));
  bool* exist_surv = (bool*) calloc(adj.rows, sizeof(bool));

  double time_spent = 0.0;

  for (int i=0; i<NB_TESTS; i++){
    copy_sparse(&host_adj, &adj);
    copy_sparse(&host_surv, &surv);
    memset(exist_surv, 0, adj.rows*sizeof(bool));

    clock_t begin = clock();
    h_mies(host_adj, host_surv, mins, exist_surv);
    clock_t end = clock();

    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;

    if (i == NB_TESTS-1)
      verify_mies(host_adj, host_surv);

    delete_sparse(&host_adj);
    delete_sparse(&host_surv);
  }

  free(mins);
  free(exist_surv);

  time_spent /= NB_TESTS;
  return time_spent;

}



double run_gpu( const sparse_mat & adj, const sparse_mat & surv )
{
  std::cout << " Compute MIES on GPU..." << std::endl;

  sparse_mat host_adj;
  sparse_mat host_surv;

  //// Warm up with an instance
   copy_sparse(&host_adj, &adj);
   copy_sparse(&host_surv, &surv);
   d_mies(host_adj, host_surv);
   delete_sparse(&host_adj);
   delete_sparse(&host_surv);
  ////
  
  double time_spent = 0.0;

  for (int i=0; i<NB_TESTS; i++){
    copy_sparse(&host_adj, &adj);
    copy_sparse(&host_surv, &surv);

    d_mies(host_adj, host_surv);

    time_spent += GPU_ELAPSED_TIME;

    if (i == NB_TESTS-1)
      verify_mies(host_adj, host_surv);

    delete_sparse(&host_adj);
    delete_sparse(&host_surv);
  }

  time_spent /= NB_TESTS;
  return time_spent/1000.0;
}



int main( int argc, char* argv[] ){

  sparse_mat* adj = NULL;
  sparse_mat* surv= NULL;

  /* ====== Generate the data ====== */

  std::cout << " Generate graphs..." << std::endl;
  generate_graphs(&adj, &surv);
  //print_sparse_py(*adj);

  std::cout << "    " << adj->rows << " nodes" << std::endl;
  std::cout << "    " << adj->nnz  << " edges" << std::endl;
  std::cout << std::endl;

  /* ====== On CPU ====== */

  double cpu_time = run_cpu( *adj, *surv);
  std::cout << "   " << cpu_time << " s" << std::endl << std::endl;


  /* ====== On GPU ====== */

  double gpu_time = run_gpu( *adj, *surv );
  std::cout << "   " << gpu_time << " s" << std::endl << std::endl;

  delete_sparse(adj);  free(adj);
  delete_sparse(surv); free(surv);
  
  return 0;
}

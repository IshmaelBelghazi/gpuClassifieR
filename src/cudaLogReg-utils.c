#include <Rmath.h>

#include "cudaLogReg.h"
#include "cudaLogReg-utils.h"
// This function does no allow for inplace summation
void _sum_matrices(const double * restrict A, const double * restrict B,
                   const double alpha, double * restrict C,
                   const int nrow, const int ncol) {

  for (int j = 0; j < ncol; j++) {
    for (int i = 0; i < nrow; i++) {
      C[idx(i, j, nrow)] = A[idx(i, j, nrow)] + alpha * B[idx(i, j, nrow)];
    }
  }
}

double _log_sum_exp(const double * restrict array, const int ar_size) {

  double sum = 0.0;
  double array_max = array[0];

  /* Getting array maximum */
  for (int i = 1; i < ar_size; i++) {
    if (array[i] > array_max) {
      array_max = array[i];
    }
  }
  /* computing exponentials sum */
  for (int i = 0; i < ar_size; i++) {
    sum += exp(array[i] - array_max);
  }
  return array_max + log(sum);
}

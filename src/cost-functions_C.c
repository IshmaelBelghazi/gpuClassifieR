#include <R.h>
#include <R_ext/BLAS.h>

#include "cudaLogReg.h"
#include "cudaLogReg-utils.h"
#include "cost-functions_C.h"

double _get_cross_entropy(const double * log_Y, const int * dim_log_Y,
                          const double * T, const int * dim_T) {

  if (dim_T[0] != dim_log_Y[0]) error("dimensions mismatch 1"); // T is K X N.
  int K = dim_T[0];
  if (dim_T[1] != dim_log_Y[1]) error("dimensions mismatch 2"); // log_Y is K X N
  int N = dim_T[1];

  int one = 1;
  double cost = 0;

  for (int j = 0; j < N; j++) {
    // should I use daxpy? ddot does not scale as well, especially memory-wise,
    // when running out of cache (Which is bound to happen
    // for big data sets this package aims to handle.).
    cost -= F77_CALL(ddot)(&K, &log_Y[j * K], &one, &T[j * K], &one);
  }
  return (cost / (N * K));

}

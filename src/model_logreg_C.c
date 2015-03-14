#include <R.h>
#include <R_ext/BLAS.h>

#include "cudaLogReg.h"

#include "cudaLogReg-utils.h"
#include "cost-functions_C.h"
#include "model_logreg_C.h"

void _set_condprob_logreg(const double * restrict X, const int * restrict dim_X,
                          const double * restrict W, const int * restrict dim_W,
                          double * restrict Y, const int * restrict dim_Y,
                          const int normalize, const int log_domain) {

  if (dim_X[0] != dim_Y[1]) error("dimensions mismatch");  // X is N X M
  int N = dim_X[0];
  if (dim_X[1] != dim_W[1]) error("dimensions mismatch"); // W is K X M
  int M = dim_X[1];
  if (dim_W[0] != dim_Y[0]) error("dimensions mismatch"); // Y is K X N
  int K = dim_W[0];

  // Computing activation
  char * trans_W = "N", * trans_X = "T";
  double alpha = 1.0, beta = 0.0;
  F77_CALL(dgemm)(trans_W, trans_X, &K, &N, &M, &alpha,
                  W, &K, X, &N, &beta, Y, &K);

  // Normalizing class conditional log-probabilities
  if (normalize) {
    double log_evidence = 0;
    for (int j = 0; j < N; j++) {
      log_evidence = _log_sum_exp(&Y[j * K], K);
      // substract log_evidence from each colum to get normalized log
      // probabilities
      for (int i = 0; i < K; i++) {
        Y[idx(i, j, K)] -= log_evidence;
      }
    }
  }

  // Taking exponential of each element if not in the log_domain
  if (!log_domain) {
    for (int j = 0; j < N; j++) {
      for (int i = 0; i < K; i++) {
        Y[idx(i, j, K)] = exp(Y[idx(i, j, K)]);
      }
    }
  }
}

double _get_cost_logreg(const double * restrict X, const int * restrict dim_X,
                        const double * restrict W, const int * restrict dim_W,
                        const double * restrict T, const int * restrict dim_T,
                        const double * decay) {

  int M = dim_W[1], N = dim_T[1], K = dim_T[0];
  double * log_Y_ptr = Calloc(K * N, double); // Calloc has its own error handling

  int dim_log_Y[2] = {K, N};
  // Setting class conditional probabiltities
  _set_condprob_logreg(X, dim_X, W, dim_W, log_Y_ptr, dim_log_Y, 1, 1);
  double cost = _get_cross_entropy(log_Y_ptr, dim_log_Y, T, dim_T);
  // Adding weight decay
  int W_len = K * M, one = 1;
  double W_ssq = F77_CALL(ddot)(&W_len, W, &one, W, &one);
  cost += 0.5 * (*decay) * W_ssq;

  Free(log_Y_ptr); // Free has its own error handling

  return cost;
}

void _set_grad_logreg(const double * restrict  X, const int * restrict dim_X,
                      const double * restrict W, const int * restrict dim_W,
                      const double * restrict T, const int * restrict dim_T,
                      double * restrict grad, const int * restrict dim_grad,
                      const double * decay) {


  if ( dim_X[1] != dim_grad[1]) error("dimensions mismatch"); // X is N X M
  if (dim_grad[1] != dim_W[1]) error("dimensions mismatch"); // W is K X M
  int M = dim_X[1];
  if (dim_X[0] != dim_T[1]) error("dimensions mismatch");  // T is K X N
  int N = dim_X[0];
  if (dim_W[0] != dim_T[0]) error("dimensions mismatch");
  if (dim_T[0] != dim_grad[0]) error("dimensions mismatch"); // Grad is K X M
  int K = dim_W[0];

  double * Y_ptr = Calloc(K * N, double);
  int dim_Y[2] = {K, N};

  _set_condprob_logreg(X, dim_X, W, dim_W, Y_ptr, dim_Y, 1, 0);

  double * Y_minus_T_ptr = Calloc(K * N, double);
  _sum_matrices(Y_ptr, T, -1, Y_minus_T_ptr, K, N);
  Free(Y_ptr);

  // grad = (Y - T) X. K X N N X M = K X M
  char * trans_X = "N", * trans_Y_minus_T = "N";
  double alpha = 1.0, beta = 0.0;
  F77_CALL(dgemm)(trans_Y_minus_T, trans_X,
                  &K, &M, &N, &alpha, Y_minus_T_ptr, &K, X,
                  &N, &beta, grad, &K);
  // Adding decay to gradient
  int W_len = K * M, one = 1;
  F77_CALL(daxpy)(&W_len, decay, W, &one, grad, &one);

  Free(Y_minus_T_ptr);
}

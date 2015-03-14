#include <R.h>
#include <Rdefines.h>

#include "cudaLogReg.h"
#include "model_logreg_C.h"
#include "entrypoints-model_logreg_C.h"

// class conditional probabilities
SEXP get_condprob_logreg_(SEXP X, SEXP W, SEXP normalize, SEXP log_domain) {
  // Initializing protection counter
  int nprot = 0;
  //  Getting pointers to underlying variables
  int * normalize_ptr = LOGICAL(normalize);
  int * log_domain_ptr = LOGICAL(log_domain);
  // Getting pointers to underlying C arrays.
  double * X_ptr = REAL(X);  // N X M
  double * W_ptr = REAL(W);  // K X M
  // getting arrays underlying dimension
  int * dim_X = INTEGER(GET_DIM(X));
  int * dim_W = INTEGER(GET_DIM(W));

  int N = dim_X[0], K = dim_W[0];

  // assigining for class conditional probabilities Y. is  K X N
  SEXP Y; PROTECT(Y = allocMatrix(REALSXP, K, N)); nprot++;
  // Getting Y dimensions
  int * dim_Y = INTEGER(GET_DIM(Y));
  // Initializing Y
  memset(REAL(Y), 0.0, K * N * sizeof(double));
  // Getting pointer to Y
  double * Y_ptr = REAL(Y);
  // Computing conditional probailities
  _set_condprob_logreg(X_ptr, dim_X,
                       W_ptr, dim_W,
                       Y_ptr, dim_Y,
                       normalize_ptr[0],
                       log_domain_ptr[0]);
  UNPROTECT(nprot);
  return (Y);
}

// Cross entropy (minus log-likelihood)
SEXP get_cost_logreg_(SEXP X, SEXP W, SEXP T, SEXP decay) {
  // initializing protection counter
  int nprot = 0;
  // Getting pointers to underlying C arrays.
  double * X_ptr = REAL(X); // N X K
  double * W_ptr = REAL(W); // K X M
  double * T_ptr = REAL(T); // K X N
  double * decay_ptr = REAL(decay);

  // getting arrays underlying dimension
  int * dim_X = INTEGER(GET_DIM(X));
  int * dim_W = INTEGER(GET_DIM(W));
  int * dim_T = INTEGER(GET_DIM(T));
  // Assigning Variable for cross-entropy
  SEXP cost; PROTECT(cost = allocVector(REALSXP, 1)); nprot++;
  //  Initializing cross entropy
  memset(REAL(cost), 0.0, sizeof(double));
  //  Getting Cross_entropy pointer to underlying
  double * cost_ptr = REAL(cost);
  // Computing cross entropy
  *cost_ptr = _get_cost_logreg(X_ptr, dim_X, W_ptr,
                               dim_W, T_ptr, dim_T,
                               decay_ptr);
  UNPROTECT(nprot);
  return (cost);
}

// gradient
SEXP get_grad_logreg_(SEXP X, SEXP W, SEXP T, SEXP decay) {
  //Initializing protection counter
  int nprot = 0;
  // Getting pointers to underlying C arrays.
  double * X_ptr = REAL(X); // N X M
  double * W_ptr = REAL(W); // K X M
  double * T_ptr = REAL(T); // K X N
  double * decay_ptr = REAL(decay);

  // getting arrays underlying dimension
  int * dim_X = INTEGER(GET_DIM(X));
  int * dim_W = INTEGER(GET_DIM(W));
  int * dim_T = INTEGER(GET_DIM(T));

  int M = dim_X[1], K = dim_W[0];

  SEXP grad; PROTECT(grad = allocMatrix(REALSXP, K, M)); nprot++; // K X M
  double * grad_ptr = REAL(grad);
  memset(grad_ptr, 0.0, sizeof(double) * K * M);
  int * dim_grad = INTEGER(GET_DIM(grad));

  _set_grad_logreg(X_ptr, dim_X,
                   W_ptr, dim_W,
                   T_ptr, dim_T,
                   grad_ptr, dim_grad,
                   decay_ptr);

  UNPROTECT(nprot);
  return (grad);
}

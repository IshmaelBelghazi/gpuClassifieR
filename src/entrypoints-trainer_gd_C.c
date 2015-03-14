#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>

#include "cudaLogReg.h"
#include "entrypoints-trainer_gd_C.h"
#include "model_logreg_C.h"

SEXP train_gd_(SEXP X, SEXP W, SEXP T, SEXP decay, SEXP step_size,
               SEXP max_iter, SEXP verbose, SEXP tol) {
  // Initializing protection counter
  int nprot = 0;
  // Getting pointers to underlying C arrays.
  double * X_ptr = REAL(X);  // N X M
  double * W_ptr = REAL(W);  // K X M
  double * T_ptr = REAL(T);  // K X N

  double * decay_ptr = REAL(decay);

  double * step_size_ptr = REAL(step_size);
  int * max_iter_ptr = INTEGER(max_iter);
  int * verbose_ptr = LOGICAL(verbose);
  double * tol_ptr = REAL(tol);

  // getting arrays underlying dimension
  int * dim_X = INTEGER(GET_DIM(X));
  int * dim_T = INTEGER(GET_DIM(T));
  int M = dim_X[1], K = dim_T[0];

  // Defining variables to return and protecting them against garbage collection
  SEXP W_trained; PROTECT(W_trained = allocMatrix(REALSXP, K, M)); nprot++;
  SEXP grad; PROTECT(grad = allocMatrix(REALSXP, K, M)); nprot++;
  SEXP iter; PROTECT(iter = allocVector(INTSXP, 1)); nprot++;

  // Getting pointers to underlying array
  double * W_trained_ptr = REAL(W_trained);
  double * grad_ptr = REAL(grad);
  int * iter_ptr = INTEGER(iter);
  // Setting W_old to hold pending weights
  double * W_old_ptr = Calloc(K * M,  double);
  // copying W to W_trained
  memcpy(W_trained_ptr, W_ptr, K * M * sizeof(double));
  // Getting arrays underlying demsion
  int * dim_grad = INTEGER(GET_DIM(grad));
  int * dim_W_trained = INTEGER(GET_DIM(W_trained));

  // Preparing training loop
  double step = *step_size_ptr;
  // gettting initial cost
  double cost  = _get_cost_logreg(X_ptr, dim_X, W_trained_ptr,
                                  dim_W_trained, T_ptr, dim_T, decay_ptr);
  double cost_new = R_PosInf;

  int one = 1, grad_len = K * M;
  double alpha = 0.0, grad_norm = R_PosInf;

  int stop_condition = 0;
  *iter_ptr = 0;
  while (!stop_condition) {
    *iter_ptr += 1;
    if (*verbose_ptr) Rprintf("iteration: %d/%d\n", *iter_ptr, *max_iter_ptr);
    if (*verbose_ptr) Rprintf("step_size: %f\n", step);
    if (*verbose_ptr) Rprintf("cost: %f\n", cost);
    // storing pending weights
    memcpy(W_old_ptr, W_trained_ptr, K * M * sizeof(double));
    // Computing gradient
    _set_grad_logreg(X_ptr, dim_X, W_trained_ptr, dim_W_trained, T_ptr,
                     dim_T, grad_ptr, dim_grad, decay_ptr);
    // Updating weights
    alpha = -1.0 * step;
    F77_CALL(daxpy)(&grad_len, &alpha, grad_ptr, &one, W_trained_ptr, &one);
    // getting new cost
    cost_new = _get_cost_logreg(X_ptr, dim_X, W_trained_ptr,
                                dim_W_trained, T_ptr, dim_T, decay_ptr);
    if (cost_new < cost) {
      step *= 1.1;
      cost = cost_new;
    } else {
      step *= 0.5;
      memcpy(W_trained_ptr, W_old_ptr, K * M * sizeof(double));
    }
    grad_norm = F77_CALL(dnrm2)(&grad_len, grad_ptr, &one);
    stop_condition = (*iter_ptr > *max_iter_ptr) || (grad_norm < *tol_ptr);
  }

  Free(W_old_ptr);

  // Creating return list to return
  SEXP results; PROTECT(results = allocVector(VECSXP, 3)); nprot++;
  SET_VECTOR_ELT(results, 0, W_trained);
  SET_VECTOR_ELT(results, 1, grad);
  SET_VECTOR_ELT(results, 2, iter);

  SEXP results_names; PROTECT(results_names = allocVector(STRSXP, 3)); nprot++;
  const char * names[3] = {"weights", "final_grad", "final_iter"};
  for (int i = 0; i < 3; i++) {
    SET_STRING_ELT(results_names, i, mkChar(names[i]));
  }
  setAttrib(results, R_NamesSymbol, results_names);

  UNPROTECT(nprot);
  return results;
}

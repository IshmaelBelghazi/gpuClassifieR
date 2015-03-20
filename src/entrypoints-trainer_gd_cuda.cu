#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>

#include "entrypoints-trainer_gd_cuda.h"
#include "model_logreg_cuda.h"
#include "cudaLogReg.h"

SEXP train_gd_cuda(SEXP X, SEXP W, SEXP T, SEXP decay, SEXP step_size,
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
  int M = dim_X[1], N = dim_X[0], K = dim_T[0];

  // Defining variables to return and protecting them against garbage collection
  SEXP W_trained; PROTECT(W_trained = allocMatrix(REALSXP, K, M)); nprot++;
  SEXP grad; PROTECT(grad = allocMatrix(REALSXP, K, M)); nprot++;
  SEXP iter; PROTECT(iter = allocVector(INTSXP, 1)); nprot++;

  // Getting pointers to underlying array
  double * W_trained_ptr = REAL(W_trained);
  double * grad_ptr = REAL(grad);
  int * iter_ptr = INTEGER(iter);

  // copying W to W_trained
  memcpy(W_trained_ptr, W_ptr, K * M * sizeof(double));

  // Declaring device array pointers
  double * dev_X_ptr;
  double * dev_W_trained_ptr;
  double * dev_W_old_ptr; // To hold pending weights
  double * dev_T_ptr;
  double * dev_grad_ptr;
  double * dev_Y_ptr;  // repetition of seemingly closely similar pointer
  // arrays is done to avoir expensive cudaMalloc/cudaFree calls
  double * dev_log_Y_ptr;
  double * dev_Y_minus_T_ptr;
  // Allocating and transfering to device
  // Initializing cublas handle
  cublasHandle_t handle;
  // Creating handle
  CUBLAS_CALL(cublasCreate(&handle));

  // allocating memory on the device
  CUDA_CALL(cudaMalloc((void **) &dev_X_ptr, N * M * sizeof(double)));
  CUDA_CALL(cudaMalloc((void **) &dev_W_trained_ptr, K * M * sizeof(double)));
  CUDA_CALL(cudaMalloc((void **) &dev_W_old_ptr, K * M * sizeof(double)));
  CUDA_CALL(cudaMalloc((void **) &dev_T_ptr, K * N * sizeof(double)));
  CUDA_CALL(cudaMalloc((void **) &dev_Y_ptr, K * N * sizeof(double)));
  CUDA_CALL(cudaMalloc((void **) &dev_log_Y_ptr, K * N * sizeof(double)));
  CUDA_CALL(cudaMalloc((void **) &dev_Y_minus_T_ptr, K * N * sizeof(double)));
  CUDA_CALL(cudaMalloc((void **) &dev_grad_ptr, K * M * sizeof(double)));
  // Copying matrices to device
  CUBLAS_CALL(cublasSetMatrix(N, M, sizeof(double), X_ptr, N, dev_X_ptr, N));
  CUBLAS_CALL(cublasSetMatrix(K, M, sizeof(double), W_trained_ptr, K,
                              dev_W_trained_ptr, K));
  CUBLAS_CALL(cublasSetMatrix(K, N, sizeof(double), T_ptr, K, dev_T_ptr, K));

  // Preparing training loop
  double step = *step_size_ptr;
  // gettting initial cost
  double cost = 0;
  _set_cost_logreg_cuda(handle, dev_X_ptr, dev_W_trained_ptr, dev_T_ptr,
                        dev_log_Y_ptr, M, N, K, decay_ptr, &cost);

  double alpha = 0.0, beta = 1.0;

  double grad_norm = R_PosInf, cost_new = R_PosInf;
  *iter_ptr = 0;
  int stop_condition = (*iter_ptr > *max_iter_ptr) && (*max_iter_ptr >= 0);
  while (!stop_condition) {
    *iter_ptr += 1;
    if (*verbose_ptr) Rprintf("iteration: %d/%d\n", *iter_ptr, *max_iter_ptr);
    if (*verbose_ptr) Rprintf("step_size: %f\n", step);
    if (*verbose_ptr) Rprintf("cost: %f\n", cost);
    // storing pending weights
    CUDA_CALL(cudaMemcpy(dev_W_old_ptr, dev_W_trained_ptr,
                         K * M * sizeof(double), cudaMemcpyDeviceToDevice));
    // Computing gradient
    _set_grad_logreg_cuda(handle, dev_X_ptr, dev_W_trained_ptr, dev_T_ptr, dev_Y_ptr,
                          dev_Y_minus_T_ptr, M, N, K, decay_ptr, dev_grad_ptr);
    // updating weights
    alpha = -1.0 * step;
    CUBLAS_CALL(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M,
                            &alpha, dev_grad_ptr, K, &beta, dev_W_trained_ptr,
                            K, dev_W_trained_ptr, K));
    // Getting new cost
    _set_cost_logreg_cuda(handle, dev_X_ptr, dev_W_trained_ptr, dev_T_ptr,
                          dev_log_Y_ptr, M, N, K, decay_ptr, &cost_new);
    if (cost_new < cost) {
      step *= 1.1;
      cost = cost_new;
    } else {
      step *= 0.5;
      CUDA_CALL(cudaMemcpy(dev_W_trained_ptr, dev_W_old_ptr,
                           K * M * sizeof(double), cudaMemcpyDeviceToDevice));
    }
    // checking gradient norm
    CUBLAS_CALL(cublasDnrm2_v2(handle, K * M, dev_grad_ptr, 1, &grad_norm));
    stop_condition = (*iter_ptr > *max_iter_ptr &&
                      *max_iter_ptr >=0) || (grad_norm < *tol_ptr);
  }
  // transfering to host
  CUBLAS_CALL(cublasGetMatrix(K, M, sizeof(double), dev_W_trained_ptr, K, W_trained_ptr, K));
  CUBLAS_CALL(cublasGetMatrix(K, M, sizeof(double), dev_grad_ptr, K, grad_ptr, K));

  // freeing allocated memory
  cudaFree(dev_X_ptr);
  cudaFree(dev_W_trained_ptr);
  cudaFree(dev_W_old_ptr);
  cudaFree(dev_T_ptr);
  cudaFree(dev_Y_ptr);
  cudaFree(dev_Y_minus_T_ptr);
  cudaFree(dev_log_Y_ptr);
  cudaFree(dev_grad_ptr);
  cublasDestroy(handle);

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

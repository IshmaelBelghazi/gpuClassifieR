#include <R.h>
#include <Rmath.h>

#include <cuda.h>
#include <cublas_v2.h>

#include "cudaLogReg.h"

#include "model_logreg_cuda.h"

__global__ void vectorMult(double * a, const double * __restrict__ b,
                           const double alpha, int arr_len, double * c) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < arr_len) {
    c[tid] =  alpha * a[tid] * b[tid];
  }
}

__global__ void setCondProbKern(double * __restrict__ Y,  // conditonal probabilities
                                const int K,  // number of rows
                                const int N,  // number of columns
                                const int normalize, const int log_domain) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;  // thread id over all blocks
  int cache_id = threadIdx.x;  // thread id whithin each block

  extern __shared__ double exp_activation[]; // dynamically allocates shared
  // memory at runtime
  __shared__ double log_evidence;

  // TODO(ISHMAEL): Add logsumexp for numerical stability

  // taking exponential of activation function
  if (cache_id < K && tid < K * N)
    exp_activation[cache_id] = exp(Y[tid]);

  __syncthreads();

  // computing sum
  if (cache_id == 0) {
    log_evidence = 0;
    int i = 0;
    while (i < K) {
      log_evidence += exp_activation[i];
      ++i;
    }
    log_evidence = log(log_evidence);
  }

  __syncthreads();

  if (normalize)
    Y[tid] -= log_evidence;

  if (!log_domain)
    Y[tid] = exp(Y[tid]);
}

void _set_condprob_logreg_cuda(cublasHandle_t handle,
                               const double * __restrict__ dev_X,
                               const double * __restrict__ dev_W,
                               double * __restrict__ dev_Y,
                               const int M, const int N, const int K,
                               const int normalize, const int log_domain) {
  // Computing activations
  double one = 1.0, zero = 0.0;
  CUBLAS_CALL(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, N, M, &one,
                          dev_W, K, dev_X, N, &zero, dev_Y, K));

  if (normalize || !log_domain) {
    // For now N blocks of K threads, with block shared memory of size K *double
    setCondProbKern <<< N, K, K * sizeof(double)>>>(dev_Y, K, N, normalize,
        log_domain);
  }
}

void _set_cost_logreg_cuda(cublasHandle_t handle,
                           const double * __restrict__ dev_X,
                           const double * __restrict__ dev_W,
                           const double * __restrict__ dev_T,
                           double * __restrict__ dev_log_Y,
                           const int M, const int N, const int K,
                           const double * decay, double * cost) {
  // I. Computing log probabilities
  int normalize = 1, log_domain = 1;
  // TODO(Ishmael): Fix this to respect in/out C convention i.e:
  //f(in_1, ..., in_k, ...., out_1, ..., out_t)
  // I. Computing class condiytional log_probabilties
  _set_condprob_logreg_cuda(handle, dev_X, dev_W,
                            dev_log_Y, M, N, K, normalize, log_domain);
  // II. Computing cross entropy
  // II.a element wise multiplication of dev_log_Y and dev_T. Note that log_Y
  // is a log probability. Therefore each elements of the sum is positive. This allow using
  // cuBlas function to sum quickly.
  // For now use N blocks of K threads. This is only temporary!
  vectorMult <<< N, K>>>(dev_log_Y, dev_T,
                         -1.0 / (K * N), K * N, dev_log_Y);
  // II.b sum all array elements
  // CUBLAS_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  // Summing all elements
  CUBLAS_CALL(cublasDasum(handle, K * N, dev_log_Y, 1, cost));
  // III.c adding regularization
  double W_ssq = 0;
  cublasDdot_v2(handle, K * M, dev_W, 1, dev_W, 1, &W_ssq);
  *cost += *decay * 0.5 * W_ssq;
}

void _set_grad_logreg_cuda(cublasHandle_t handle,
                           const double * __restrict__ dev_X,
                           const double * __restrict__ dev_W,
                           const double * __restrict__ dev_T,
                           double * __restrict__ dev_Y,
                           double * __restrict__ dev_Y_minus_T,
                           const int M, const int N, const int K,
                           const double * decay,
                           double * dev_grad) {

  // Computing gradients
  // I computing class conditional probabilties
  int normalize = 1, log_domain = 0;
  _set_condprob_logreg_cuda(handle, dev_X, dev_W, dev_Y,
                            M, N, K, normalize, log_domain);
  // II computing Y - T
  double one = 1.0, m_one = -1.0;
  CUBLAS_CALL(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, N,
                          &one, dev_Y, K, &m_one, dev_T, K, dev_Y_minus_T, K));
  // Computing gradients
  // III.1 Multiplying data by Y_minus_T to get grad
  double zero = 0.0;
  CUBLAS_CALL(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          K, M, N, &one, dev_Y_minus_T, K, dev_X,
                          N, &zero, dev_grad, K));
  // III.2 adding decay gradient
  CUBLAS_CALL(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, &one,
                          dev_grad, K, decay, dev_W, K, dev_grad, K));



}

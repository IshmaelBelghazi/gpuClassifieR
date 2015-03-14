#ifndef CUDALOGREG
#define CUDALOGREG

#include <R_ext/PrtUtil.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/*==============\
| HELPER MACROS |
\==============*/

// Column major to row-column notation
#define idx(i, j, lda) (((j) * (lda) + (i) ))
//Takes pointer to double and dimension then
//   allocates and sets identity matrix
#define IDMAT(M, K) {                               \
    M == (double *)R_alloc(K * K, sizeof(double));  \
    memset(M, 0.0, K * K * sizeof(double));         \
    for(int i = 0; i < K; i++) M[i * K + i] = 1.0;  \
}

/*===========================\
| CUDA ERROR HANDLING MACROS |
\===========================*/

#define CUDA_CALL(x) {if((x) != cudaSuccess) {                   \
      Rprintf("CUDA error at %s:%d\n", __FILE__, __LINE__);      \
      Rprintf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
      cudaDeviceReset();                                         \
      error("CUDA call has failed\n");}}

/*=============================\
| CUBLAS ERROR HANDLING MACROS |
\=============================*/
#define CUBLAS_CALL(x) {cublasStatus_t cublas_err_status = (x);         \
    if(cublas_err_status != CUBLAS_STATUS_SUCCESS){                     \
      Rprintf("CUBLAS error at %s:%d\n", __FILE__, __LINE__);           \
      Rprintf(" %s\n", _cublasGetErrorString(cublas_err_status));       \
      cudaDeviceReset();                                                \
      error("CUBLAS call has failed\n");}}
// cuBLAS API errors
static inline const char *_cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
  }

  return "<unknown error>";
}

#endif // CUDALOGREG

#ifndef CUDALOGREG
#define CUDALOGREG

#include <R_ext/PrtUtil.h>


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

#endif // CUDALOGREG

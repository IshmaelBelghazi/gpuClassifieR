#ifndef CUDALOGREGUTILS
#define CUDALOGREGUTILS

void _sum_matrices(const double * restrict A, const double * restrict B,
                   const double alpha, double * restrict C,
                   const int nrow, const int ncol);

double _log_sum_exp(const double * restrict array, const int ar_size);

#endif

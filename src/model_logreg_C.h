#ifndef MODEL_LOGREG_C
#define MODEL_LOGREG_C

void _set_condprob_logreg(const double * restrict X, const int * restrict dim_X,
                          const double * restrict W, const int * restrict dim_W,
                          double * restrict Y, const int * restrict dim_Y,
                          const int normalize, const int log_domain);

double _get_cost_logreg(const double * restrict X, const int * restrict dim_X,
                        const double * restrict W, const int * restrict dim_W,
                        const double * restrict T, const int * restrict dim_T,
                        const double * decay);

void _set_grad_logreg(const double * restrict  X, const int * restrict dim_X,
                      const double * restrict W, const int * restrict dim_W,
                      const double * restrict T, const int * restrict dim_T,
                      double * restrict grad, const int * restrict dim_grad,
                      const double * decay);

#endif // MODEL_LOGREG_C

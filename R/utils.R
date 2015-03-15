## * Various utility functions
## ** Log Sum exp
##' Log sum exp in R
##'
##' @param x vector to log sum exp
##' @param ... pass through parameters
##' @return log sum exp
##' @author Mohamed Ishmael Diwan Belghazi
logsumexp_R <- function(x, ...) {
    max_x <- max(x, ...)
    max_x + log(sum(exp(x - max_x)))
}

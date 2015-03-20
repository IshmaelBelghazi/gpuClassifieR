## * Gradient descent trainer
.train_gd <- function(object, feats, targets, decay=NULL, step_size=NULL,
                      max_iter=NULL, verbose=FALSE, tol=1e-6,
                      backend="R", ...) {

    ## Features should be N X M. Targets should be N X K
    stopifnot(NROW(feats) == NROW(targets))
    ## Weights should be M X K
    stopifnot(NROW(object$weights) == NCOL(feats))
    stopifnot(NCOL(object$weights) == NCOL(targets))

    if(is.null(max_iter) || is.infinite(max_iter)) {
        message("no maximum iteration specified. Running to convergence.")
        max_iter <- -1
    }

    if(is.null(step_size)) {
        message("unspecified step size. defaulting to 0.01")
        step_size <- 0.01
    }
    if(is.null(decay)) {
        message("unspecified decay coeffecient. Setting it to zero")
        decay <- 0.0
    }

    switch(backend,
           "R"={
               do.call(".train_gd_R", as.list(match.call())[-1])
           },
           "C"={
               do.call(".train_gd_C", as.list(match.call())[-1])
           },
           "CUDA"={
               stop("CUDA backend not usable in C only branch")
           },
       {
           stop("unrecognized computation backend")
       })
}

## * Gradient descent trainers backends
## ** Gradient descent trainers: R Backend

.train_gd_R <- function(object, feats, targets, decay=NULL , step_size=NULL,
                        max_iter=NULL, verbose=FALSE, tol=1e-6, ...) {

    ## TODO(Ishmael): Add passthrough option for computation backend
    backend <- "R"

    ## initializing training variables
    cost <- get_cost(object, feats, targets, decay, backend)
    iter <- 0
    stop_condition <- (iter > max_iter && max_iter >=0)
    while(!stop_condition) {
        iter <- iter + 1
        if (verbose) message(sprintf("iteration: %d/%d", iter, max_iter))
        if (verbose) message(sprintf("stepsize: %f", step_size))
        if (verbose) message(sprintf("cost:%f", cost))
        weights_old <- coef(object)
        grad <- get_grad(object, feats, targets, decay, backend)
        object$weights <- coef(object) - step_size * grad
        cost_new <- get_cost(object, feats, targets, decay, backend)
        if (cost_new < cost) {
            step_size <- step_size * 1.1
            cost <- cost_new
        } else {
            step_size <- 0.5 * step_size
            object$weights <- weights_old
        }
        stop_condition <- (iter > max_iter &&
                               max_iter >= 0) || (norm(grad, type="F") < tol)
    }

    object$final_grad <- grad
    object$final_iter <- iter
    return(object)
}

## ** Gradient descent trainers: C Backend
##' @useDynLib gpuClassifieR, .registration=TRUE
.train_gd_C <-  function(object, feats, targets, decay=NULL, step_size=NULL,
                        max_iter=NULL, verbose=FALSE, tol=1e-6, ...) {

    weights <- coef(object)
    results <- .Call("train_gd_", as.matrix(feats),
                     t(as.matrix(weights)),  ## Transpose to minimize cache
                     t(as.matrix(targets)),  ## misuse during BLAS operations
                     as.double(decay),
                     as.double(step_size),
                     as.integer(max_iter),
                     as.logical(verbose),
                     as.double(tol))

    object$weights <- t(results$weights)
    object$final_grad <- results$final_grad
    object$final_iter <- results$final_iter
    return(object)
}

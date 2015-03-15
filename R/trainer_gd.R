## * Generics
##' Gradient Descent trainer generic
##'
##' @param object model.spec object
##' @param feats features (N X M)
##' @param targets targets (N X K)
##' @param decay L2 regularisation coefficient
##' @param step_size gradient descent step size
##' @param max_iter Maximum iteration
##' @param verbose prints output
##' @param tol assume convergence when norm of gradient is smaller than tol
##' @param backend Computation backend ("R", "C", or "CUDA")
##' @param ... any other passthrough parameters
##' @return model.spec object
##' @author Mohamed Ishmael Diwan Belghazi
train <- function(object, feats, targets, decay=NULL, step_size=NULL, max_iter=NULL,
                  verbose=FALSE, tol=1e-6, backend="R", ...) {
    UseMethod("train")
}

train.default <- function(object, feats, targets, decay=NULL, step_size=NULL, max_iter=NULL,
                  verbose=FALSE, tol=1e-6, backend="R", ...) {
    stop("unkown object class")
}
##' @describeIn train Gradient descent trainer
train.model.spec <- function(object, feats, targets, decay=NULL, step_size=NULL, max_iter=NULL,
                             verbose=FALSE, tol=1e-6, backend="R", ...) {

    ## Features should be N X M. Targets should be N X K
    stopifnot(NROW(feats) == NROW(targets))
    ## Weights should be M X K
    stopifnot(NROW(object$weights) == NCOL(feats))
    stopifnot(NCOL(object$weights) == NCOL(targets))

    if(is.null(max_iter)) {
        message("no maximum iteration specified. Running until convergence.")
        max_iter <- Inf
    }

    if(is.null(step_size)) {
        message("unspecified step size. defaulting to 0.01")
        step_size <- 0.01
    }
    if(is.null(max_iter)) {
        message("no maximum iteration specified. Running until convergence.")
        max_iter <- Inf
    }
    if(is.null(decay)) {
        message("unspecified decay coeffecient. Setting it to zero")
        decay <- 0.0
    }

    switch(backend,
           "R"={
               message("R backend")
               do.call("train_gd_R", as.list(match.call())[-1])
           },
           "C"={
               message("C backend")
               do.call("train_gd_C", as.list(match.call())[-1])

           },
           "CUDA"={
               message("CUDA backend")
               do.call("train_gd_CUDA", as.list(match.call())[-1])
           },
       {
           stop("unrecognized computation backend")
       })
}

## * Gradient descent trainers
## ** Gradient descent trainers: R Backend

train_gd_R <- function(object, feats, targets, decay=NULL , step_size=NULL, max_iter=NULL,
                       verbose=FALSE, tol=1e-6, ...) {

    ## TODO(Ishmael): Add passthrough option for computation backend
    backend <- "R"

    ## initializing training variables
    cost <- get_cost(object, feats, targets, decay, backend)
    stop_condition <- FALSE
    iter <- 0
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
        stop_condition <- (iter > max_iter) || (norm(grad, type="F") < tol)
    }

    object$final_grad <- grad
    object$final_iter <- iter
    return(object)
}

## ** Gradient descent trainers: C Backend
train_gd_C <-  function(object, feats, targets, decay=NULL, step_size=NULL,
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

## ** Gradient descent trainers: CUDA Backend
train_gd_CUDA <-  function(object, feats, targets, decay=NULL, step_size=NULL,
                           max_iter=NULL, verbose=FALSE, tol=1e-6, ...) {

    weights <- coef(object)
    results <- .Call("train_gd_cuda", as.matrix(feats),
                     t(as.matrix(weights)),
                     t(as.matrix(targets)),
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

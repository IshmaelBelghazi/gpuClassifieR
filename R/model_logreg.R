##' Computes Logistic regression class conditional probabilities
##'
##' @param feats features. (N feats M)
##' @param weights model weights. (M feats K)
##' @param normalize Normalize probabilties
##' @param log_domain Compute log-probabilties
##' @param backend Computation backend ("R", "C", or "CUDA")
##' @return Class conditional probabilities
##' @author Mohamed Ishmael Diwan Belghazi
##' @useDynLib cudaLogReg, .registration=TRUE
get_condprob_logreg <- function(feats, weights, normalize=TRUE,
                                log_domain=FALSE, backend="R") {

    switch(backend,
           "R"={
               condprob <- feats %*% weights
               ## Normalizing probabilities
               if (normalize) {
                   condprob <- t(apply(condprob, 1,
                                       FUN = function(feats) feats - logsumexp_R(feats)))
               }
               if (!log_domain) condprob <- exp(condprob)
           },
           "C"={
               condprob <- t(.Call("get_condprob_logreg_",
                                   as.matrix(feats),
                                   t(weights),
                                   as.logical(normalize),
                                   as.logical(log_domain)))
           },
           "CUDA"={
               condprob <- t(.Call("get_condprob_logreg_cuda",
                                   as.matrix(feats),
                                   t(weights),
                                   as.logical(normalize),
                                   as.logical(log_domain)))
           },
       {
           stop("unrocognized computation bckend")
       })
    return(condprob)
}

##' Computes cost function for logistic regression
##'
##' @param feats Features (N feats M)
##' @param weights Model weights (M feats K)
##' @param targets Targets (N feats K)
##' @param decay L2 regularization decay factor
##' @param backend Computation backend ("R", "C", or "CUDA")
##' @return logistic regression cost (cross-entropy)
##' @author Mohamed Ishmael Diwan
##' @useDynLib cudaLogReg, .registration=TRUE
get_cost_logreg <- function(feats, weights, targets, decay=0.0, backend="R") {


    switch(backend,
           "R"={
               log_prob <- get_condprob_logreg(feats, weights, log_domain=TRUE, backend="R")
               cost <- -mean(log_prob * targets) + 0.5 * decay * sum(weights^2)
           },
           "C"={
               cost <- .Call("get_cost_logreg_", as.matrix(feats),
                             t(as.matrix(weights)),
                             t(as.matrix(targets)),
                             as.double(decay))
           },
           "CUDA"={
               cost <- .Call("get_cost_logreg_cuda", as.matrix(feats),
                             t(as.matrix(weights)),
                             t(as.matrix(targets)),
                             as.double(decay))
           },
       {
           stop("unrecognized computation backend")
       })
    return(cost)
}


##' Computes gradient for logistic regression model
##'
##' @param feats Features. (N feats M)
##' @param weights Model weights (M feats K)
##' @param targets One-hot encoded targets (N feats K)
##' @param decay L2 regularization decay factor
##' @param backend Computation backend ("R", "C", or "CUDA")
##' @return gradient (M feats K)
##' @author Mohamed Ishmael Diwan Belghazi
##' @useDynLib cudaLogReg, .registration=TRUE
get_grad_logreg <- function(feats, weights, targets, decay=0.0, backend = "R") {

    switch(backend,
           "R"={
               prob <- get_condprob_logreg(feats, weights, log_domain=FALSE, backend="R")
               grad <-  (t(feats) %*% (prob - targets)) + decay * weights
           },
           "C"={
               grad <- t(.Call("get_grad_logreg_",
                               as.matrix(feats),
                               t(as.matrix(weights)),
                               t(as.matrix(targets)),
                               as.double(decay)))
           },
           "CUDA"={
               grad <- t(.Call("get_grad_logreg_cuda",
                               as.matrix(feats),
                               t(as.matrix(weights)),
                               t(as.matrix(targets)),
                               as.double(decay)))
           },
       {
           stop("unrecognized computation backend")
       })
    return(grad)
}

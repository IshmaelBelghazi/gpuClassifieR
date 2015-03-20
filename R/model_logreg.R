## * Class conditional probabilities
##' @useDynLib gpuClassifieR, .registration=TRUE
.get_condprob <- function(feats, weights, normalize=TRUE,
                                log_domain=FALSE, backend='R') {

    switch(backend,
           R={
               condprob <- feats %*% weights
               ## Normalizing probabilities
               if (normalize) {
                   condprob <- t(apply(condprob, 1,
                                       function(feats) feats - .logsumexp_R(feats)))
               }
               if (!log_domain) condprob <- exp(condprob)
           },
           C={
               condprob <- t(.Call('get_condprob_logreg_',
                                   as.matrix(feats),
                                   t(weights),
                                   as.logical(normalize),
                                   as.logical(log_domain)))
           },
           CUDA={
               stop("CUDA backend not usable in C only branch")
           },
       {
           stop('unrecognized computation bckend')
       })
    return(condprob)
}
## * Cost
##' @useDynLib gpuClassifieR, .registration=TRUE
.get_cost <- function(feats, weights, targets, decay=0.0, backend='R') {


    switch(backend,
           R={
               log_prob <- .get_condprob(feats, weights, log_domain=TRUE,
                                         backend='R')
               cost <- -mean(log_prob * targets) + 0.5 * decay * sum(weights^2)
           },
           'C'={
               cost <- .Call('get_cost_logreg_', as.matrix(feats),
                             t(as.matrix(weights)),
                             t(as.matrix(targets)),
                             as.double(decay))
           },
           CUDA={
               stop("CUDA backend not usable in C only branch")
           },
       {
           stop('unrecognized computation backend')
       })
    return(cost)
}

## * Cost gradient
##' @useDynLib gpuClassifieR, .registration=TRUE
.get_grad <- function(feats, weights, targets, decay=0.0, backend='R') {

    switch(backend,
           R={
               prob <- .get_condprob(feats, weights, log_domain=FALSE,
                                     backend='R')
               grad <-  (t(feats) %*% (prob - targets)) + decay * weights
           },
           C={
               grad <- t(.Call('get_grad_logreg_',
                               as.matrix(feats),
                               t(as.matrix(weights)),
                               t(as.matrix(targets)),
                               as.double(decay)))
           },
           CUDA={
               stop("CUDA backend not usable in C only branch")
           },
       {
           stop('unrecognized computation backend')
       })
    return(grad)
}

## * Prediction
.predict_class <- function(condprob, backend='R') {
    switch(backend,
           R={
               predictions <- mat.or.vec(NROW(condprob), NCOL(condprob))
               max_idx <- max.col(condprob, ties.method = 'first')
               mapply(function(i, j) predictions[i, j] <<- 1.0,
                      1:NROW(condprob), max_idx)
           },
           C={
               stop(paste(backend,' backend function not implemented'))
           },
           CUDA={
               stop(paste(backend,' backend function not implemented'))
           },
       {
           stop('unrecognized computation backend')
       })

        return(predictions)
}

## * Misclassification rate
.get_error <- function(predictions, targets, backend='R') {

    switch(backend,
           R={
               mis_rate <- 1 - mean(rowSums(predictions * targets))
           },
           C={
               stop(paste(backend,' not implemented'))
           },
           CUDA={
               stop(paste(backend,' not implemented'))
           },
       {
           stop('unrecognized computation backend')
       })
    return(mis_rate)
}

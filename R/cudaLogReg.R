#' cudaLogReg: a GPU linear classfier for R
#'
#' This packages implements a linear classifier with a batch gradient descent trainer.
#' Three backend are offered: R, C(BLAS), and CUDA.
#'
#' @name cudaLogReg
#' @docType package
NULL

##' Subset of MNIST
##'
##' A dataset containing a subset of thr MNIST database. The MNIST databse is a
##' large collection of handwritten digits.
##' @keywords datasets
##' @name mini_mnist
##' @docType data
##' @usage data(mini_mnist)
##' @format 2 Matrices of 1000 rows and 784 columns for train and test images
##' and 2 matrices of 1000 rows and 10 columns for the train and test targets
##' (in onehot encoding).
NULL

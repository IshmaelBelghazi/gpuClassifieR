#' gpuClassifieR: a GPU linear classifier for R
#'
#' This packages implements a linear classifier with a batch gradient descent trainer.
#' Three backend are offered: R, C, and CUDA.
#'
#' @name gpuClassifieR
#' @docType package
NULL

##' Subset of MNIST
##'
##' A dataset containing a subset of the MNIST database. The MNIST database is a
##' large collection of handwritten digits.
##' @keywords datasets
##' @name mini_mnist
##' @docType data
##' @usage data(mini_mnist)
##' @format 2 Matrices of 1000 rows and 784 columns for train and test images
##' and 2 matrices of 1000 rows and 10 columns for the train and test targets
##' (in one-hot encoding).
NULL

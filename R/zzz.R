## * Cleanup
.onUnload <- function(libpath) {
    library.dynam.unload("cudaLogReg", libpath)
}

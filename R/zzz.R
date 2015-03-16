## * Cleanup
.onUnload <- function(libpath) {
    library.dynam.unload("gpuClassifieR", libpath)
}

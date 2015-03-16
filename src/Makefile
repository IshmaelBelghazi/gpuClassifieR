include config.mk

CUDA_EXT := cu
CUDA_OBJS := entrypoints-model_logreg_cuda.o model_logreg_cuda.o entrypoints-trainer_gd_cuda.o

C_EXT := c
C_OBJS := entrypoints-model_logreg_C.o model_logreg_C.o cost-functions_C.o cudaLogReg-utils.o entrypoints-trainer_gd_C.o
R_HOME := $(shell R RHOME)

#compiler/preprocessor options
INCS := -I. -I"$(CUDA_INC)" -I"$(R_INC)"
PARAMS := $(DEVICEOPTS) -Xcompiler $(CPICFLAGS)

#linker options
LD_PARAMS := $(DEVICEOPTS) -Xlinker '$(RPATH) $(R_FRAMEWORK)'
LIBS :=  -L"$(R_LIB)" -L"$(CUDA_LIB)" -lcublas $(shell R CMD config BLAS_LIBS)

TARGETS := cudaLogReg.so

NVCC := $(CUDA_HOME)/bin/nvcc -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30, -gencode arch=compute_32,code=sm_32 -gencode arch=compute_35,code=sm_35, -gencode arch=compute_37,code=sm_37, -gencode arch=compute_50,code=sm_50
##NVCC := $(CUDA_HOME)/bin/nvcc -gencode arch=compute_35,code=sm_35

all: $(TARGETS)

$(TARGETS): $(CUDA_OBJS) $(C_OBJS)
	$(NVCC) -shared $(LD_PARAMS) $(LIBS) $(CUDA_OBJS) $(C_OBJS) -o $@

$(CUDA_OBJS): %.o: %.$(CUDA_EXT)
	$(NVCC) -c $(INCS) $(PARAMS) $^ -o $@

## TODO(Ishmael): Fine tune this. give option for gcc.
$(C_OBJS): %.o: %.$(C_EXT)
	$(NVCC) --compiler-options "-std=gnu99" -c $(INCS) $(PARAMS) $^ -o $@

veryclean: clean
	rm -rf *.so
clean:
	rm -rf *o
.PHONY: all clean veryclean

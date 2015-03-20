###########################
## ENVIRONMENT VARIABLES ##
###########################

##,-------------------------------
##| General Environement variables
##`-------------------------------

OS=$(shell uname -s)
ARCH=$(shell uname -m)

##,------------------------
##| R Environment variables
##`------------------------

## R config
R_COMMAND=R --no-save --slave
R_CONF=R --vanilla CMD config
R_getconf=$(shell $(R_CONF) $(1))
## R_HOME
R_HOME?=$(shell R RHOME)
## R include path
R_INCLUDE?=-I$(R_HOME)/include
## R library path
R_LIB?=-L$(R_HOME)/lib

##,-------------------------
##| C environement variables
##`-------------------------

## C files extension
C_EXT=.c
## C standard
C_STD=-std=gnu99
## Shared library compilation flags
CPICFLAGS?=$(call R_getconf, CPICFLAGS)
## BLAS libs
BLAS_LIBS=$(call R_getconf, BLAS_LIBS)
## Shared lib extension
DYLIB_EXT=$(call R_getconf, DYLIB_EXT)
## Dynamic Library extension
SHLIB_EXT=$(call R_getconf, SHLIB_EXT)

##,--------------------------
##| CUDA Environment variable
##`--------------------------

## Cuda file extension
CUDA_EXT=.cu
## Cuda home path
CUDA_HOME?=/usr/local/cuda
## Cuda compiler path
NVCC?=$(CUDA_HOME)/bin/nvcc
## Cuda Include path
CUDA_INCLUDE?=$(CUDA_HOME)/include
## Cuda library path
ifeq ($(ARCH), i386)
	CUDA_LIB?=$(CUDA_HOME)/lib
else
	CUDA_LIB?=$(CUDA_HOME)/lib64
endif
## Target gpu architectures
GPU_ARCH= \
    -gencode arch=compute_20,code=sm_20 \
    -gencode arch=compute_30,code=sm_30 \
    -gencode arch=compute_32,code=sm_32 \
    -gencode arch=compute_35,code=sm_35 \
    -gencode arch=compute_37,code=sm_37 \
    -gencode arch=compute_50,code=sm_50
# GPU_ARCH=-gencode arch=compute_35,code=sm_35
## MacOSX specific configuration
## Source: https://github.com/nullsatz/gputools/blob/master/src/config.mk
ifeq ($(OS), Darwin)
    ifeq ($(ARCH), x86_64)
        DEVICEOPTS?=-m64
    endif
    CUDA_LIB?=-L$(CUDA_HOME)/lib
    R_FRAMEWORK?=-F$(R_HOME)/.. -framework R
    RPATH?=-rpath $(CUDA_LIB)
endif

###########################
## COMPILATION VARIABLES ##
###########################

##,-----------------
##| Setting compiler
##`-----------------

INCLUDES=-I. -I$(CUDA_INCLUDE) $(R_INCLUDE)
PARAMS= $(DEVICEOPTS) -Xcompiler $(CPICFLAGS) --use_fast_math

##,---------------
##| Setting linker
##`---------------

LD_PARAMS=$(DEVICEOPTS) -Xlinker $(RPATH) $(R_FRAMEWORK)
LIBS=$(R_LIB) -L$(CUDA_LIB) -lcublas $(BLAS_LIBS)

##,------------------------
##| Setting build variables
##`------------------------

## SOURCES
C_SRC=$(wildcard ./*$(C_EXT))
CUDA_SRC=$(wildcard ./*$(CUDA_EXT))
## Objects
C_OBJS=$(C_SRC:$(C_EXT)=.o)
CUDA_OBJS=$(CUDA_SRC:$(CUDA_EXT)=.o)
## Targets
TARGETS:=gpuClassifieR$(DYLIB_EXT)


#################
## BUILD RULES ##
#################

all: $(TARGETS)

$(TARGETS): $(CUDA_OBJS) $(C_OBJS)
	$(NVCC) $(GPU_ARCH) -shared $(LD_PARAMS) $(LIBS) $(CUDA_OBJS) $(C_OBJS) -o $@

$(CUDA_OBJS): %.o: %$(CUDA_EXT)
	$(NVCC) $(GPU_ARCH) -c $(INCLUDES) $(PARAMS) $^ -o $@

$(C_OBJS): %.o: %$(C_EXT)
	$(NVCC) --compiler-options $(C_STD) -c $(INCS) $(PARAMS) $^ -o $@

veryclean: clean
	rm -rf *.so

clean:
	rm -rf *o

cleanup: clean

.PHONY: all clean veryclean cleanup

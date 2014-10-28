# Please note that SUFFIXES must be set with .cu before including cuda.am
# ortherwise this Makefile is not usable
SUFFIXES = .cu
include $(top_srcdir)/am/cuda.am

# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in 

AUTOMAKE_OPTIONS = subdir-objects

# enable timing measuments
TIMING_FLAGS =
if DO_TIMING
TIMING_FLAGS += -DDO_TIMING
endif

# default CPU flags
AM_CXXFLAGS = -I$(srcdir) $(CUDA_CFLAGS) $(TIMING_FLAGS)
AM_CFLAGS   = $(CUDA_CFLAGS) $(TIMING_FLAGS)
AM_CPPFLAGS = -I$(srcdir) $(CUDA_CFLAGS) $(TIMING_FLAGS)
AM_LDFLAGS  = 
AM_FCFLAGS  = -fno-underscoring

# default CUDA nvcc flags
NVCCFLAGS  += --ptxas-options -v
NVCCFLAGS  += --compiler-options -I$(srcdir) 
NVCCFLAGS += --compiler-options -fno-strict-aliasing
NVCCFLAGS += --compiler-options -Wno-unused-function
NVCCFLAGS += --compiler-options -Wno-unused-parameter
if DO_TIMING
NVCCFLAGS += --compiler-options -DDO_TIMING
endif

# if MPI is enabled, then modify default nvcc flags
if USE_MPI
NVCCFLAGS  += $(addprefix --compiler-options , $(MPI_CXXFLAGS) )
endif

# ===================================================
# PROGRAMS
# ===================================================
bin_PROGRAMS  = 

if USE_MPI
bin_PROGRAMS += testHelloMpiCuda 

#
# testHelloMpiCuda
#
testHelloMpiCuda_SOURCES  = testHelloMpiCuda.cu
# add dummy C++ source to cause C++ linking.
nodist_EXTRA_testHelloMpiCuda_SOURCES = dummy.cxx
testHelloMpiCuda_CPPFLAGS = $(AM_CPPFLAGS) $(MPI_CXXFLAGS)
testHelloMpiCuda_LDFLAGS  = $(AM_LDFLAGS)  $(MPI_LDFLAGS) $(CUDA_LIBS) 

endif
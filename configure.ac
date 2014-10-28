# ------------------------------------------------------------------------------
# Package setup: template configure.ac for a CUDA+MPI project
# ------------------------------------------------------------------------------
AC_INIT([test_MPI_CUDA], [1.0.0], [pierre.kestener@cea.fr])

AM_INIT_AUTOMAKE([-Wall -Werror foreign -Wno-portability])

# ------------------------------------------------------------------------------
# Default flags reinitialization
# ------------------------------------------------------------------------------
if test -z "$FCFLAGS"
then
    FCFLAGS="-Wall "
fi

if test -z "$CFLAGS"
then
   CFLAGS=" "
fi

if test -z "$CXXFLAGS"
then
   CXXFLAGS="-Wall "
fi

# ------------------------------------------------------------------------------
# Dummy headline formater
# ------------------------------------------------------------------------------
AC_DEFUN([CONFIGURE_HEADLINE],
[
        echo; echo "+++ $1 +++"
])


# ------------------------------------------------------------------------------
# Checks for tool support
# ------------------------------------------------------------------------------
AC_PROG_FC
AC_PROG_CXX
AC_PROG_LIBTOOL

# some compiler vendor specific flags (intel icc does not understand -Wextra)
AX_COMPILER_VENDOR
case "${ax_cv_c_compiler_vendor}" in
gnu)
	CXXFLAGS+=" -Wall -Wextra"
	;;
intel)
	# disable remark 981: operands are evaluated in unspecified order
	CXXFLAGS+=" -Wall -wd981"
	;;
*)
	;;
esac

# ------------------------------------------------------------------------------
# Checks for Fortran support
# ------------------------------------------------------------------------------
AC_LANG([Fortran])
AC_FC_LIBRARY_LDFLAGS

# ------------------------------------------------------------------------------
# Checks for C/C++ support
# ------------------------------------------------------------------------------
AC_LANG([C])
AC_LANG([C++])

# ------------------------------------------------------------------------------
# Enable silent build rules by default
# ------------------------------------------------------------------------------
m4_ifdef([AM_SILENT_RULES],[AM_SILENT_RULES([yes])],[AC_SUBST([AM_DEFAULT_VERBOSITY], [1])])

# ------------------------------------------------------------------------------
# Provides debug and release compilation modes
# ------------------------------------------------------------------------------
AC_ARG_ENABLE([debug],
   [  --enable-debug    Turn on debugging],
   [case "${enableval}" in
      yes) DEBUG=true ;;
      no)  DEBUG=false ;;
      *)   AC_MSG_ERROR([bad value ${enableval} for --enable-debug]) ;;
   esac],
   [DEBUG=false]
)
if test x$DEBUG = xtrue
then
    FCFLAGS="-g3 -O0 ${FCFLAGS}"
    CXXFLAGS="-g3 -O0 -D_GLIBCXX_DEBUG ${CXXFLAGS}"
    CFLAGS="-g3 -O0 -D_GLIBCXX_DEBUG ${CFLAGS}"
else
    FCFLAGS="-O3 ${FCFLAGS}"
    CXXFLAGS="-O3 ${CXXFLAGS}"
    CFLAGS="-O3 ${CFLAGS}"
fi

# ------------------------------------------------------------------------------
# Turn on gprof flags
# ------------------------------------------------------------------------------
AC_ARG_ENABLE([gprof],
	[  --enable-gprof    Turn on profiling with gprof (pass gnu or intel as argument)],
	[case "${enableval}" in
	      gnu | yes) 
	      	   gprof_enabled=true 
		   gprof_compiler="gnu compiler"
		   gprof_flags="-pg"
		   ;;
	      intel)
		   gprof_enabled=true
		   gprof_compiler="intel compiler"
		   gprof_flags="-p"
		   ;;
	      no)  gprof_enabled=false 
	      	   gprof_compiler="none"
		   gprof_flags=""
		   ;;
	      *) AC_MSG_ERROR([bad value ${enableval} for --enable-gprof (only yes, gnu or intel are possible)]) ;;
	 esac],
	 [gprof_enabled=false
	  gprof_compiler="none"
	  gprof_flags=""
	 ]	      
)
if test x$gprof_enabled = xtrue
then
	CXXFLAGS="${CXXFLAGS} ${gprof_flags}"
	CPPFLAGS="${CPPFLAGS} ${gprof_flags}"
	LDFLAGS="${LDFLAGS} ${gprof_flags}"
fi

# ------------------------------------------------------------------------------
# Turn on timing measurements
# ------------------------------------------------------------------------------
AC_ARG_ENABLE([timing],
	[  --enable-timing    Turn on timing measurement and report],
	[case "${enableval}" in
	      yes) 
	      	   timing_enabled=true 
		   ;;
	      no)  timing_enabled=false 
		   ;;
	      *) AC_MSG_ERROR([bad value ${enableval} for --enable-timing]) ;;
	 esac],
	 [timing_enabled=false
	 ]	      
)
AM_CONDITIONAL(DO_TIMING, test "$timing_enabled" = "true")

# ------------------------------------------------------------------------------
# Turn on double precision
# ------------------------------------------------------------------------------
AC_ARG_ENABLE([double],
	[  --enable-double    Turn on double precision computation],
	[case "${enableval}" in
	      yes) DOUBLE=yes ;;
	      no)  DOUBLE=no ;;
	      *)   AC_MSG_ERROR([bad value ${enableval} for --enable-double]) ;;
	      esac],
   	[DOUBLE=no]
)
AM_CONDITIONAL(USE_DOUBLE_PRECISION, test "$DOUBLE" = "yes")

# ------------------------------------------------------------------------------
# Detects CUDA
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ CUDA support ])
AX_CUDA

# ------------------------------------------------------------------------------
# Nvcc flags setup (remember that default NVCCFLAGS are set by AX_CUDA macro)
# ------------------------------------------------------------------------------
if test x$DEBUG = xtrue
then
    NVCCFLAGS+=" -g -G"
else
    NVCCFLAGS+=" -O3"
fi

if test x$gprof_enabled = xtrue
then
    NVCCFLAGS+=" --compiler-options ${gprof_flags}"
fi

# ------------------------------------------------------------------------------
#    Check for lib GraphicsMagick++
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ Graphics Magick support ])
PKG_CHECK_MODULES(GM, GraphicsMagick++ >= 1.1, GM_PRESENT='yes',GM_PRESENT='no')
AC_SUBST([GM_CFLAGS])
AC_SUBST([GM_LIBS])
AM_CONDITIONAL(HAVE_GM, test "$GM_PRESENT" = "yes")

# ------------------------------------------------------------------------------
#    Check for lib MPI (OpenMPI or MPICH)
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ MPI compiler ])
AC_ARG_ENABLE(mpi, [AC_HELP_STRING([--enable-mpi],[MPI library required])], enable_mpi=$enableval, enable_mpi=no)

if test "$enable_mpi" = "yes"; then
   ACX_MPI([],[AC_MSG_ERROR([could not find mpi library for --enable-mpi])])
   AC_CHECK_PROG(MPIRUN, mpirun, mpirun)
   AC_SUBST(MPIRUN)
else
   AC_MSG_NOTICE([MPI is currently disabled; use option --enable-mpi to enable])
fi
AM_CONDITIONAL(USE_MPI, test "$enable_mpi" = "yes")

# try to find if we are using OpenMPI / MPICH by looking inside mpi.h
AS_IF([test "$enable_mpi" = "yes"],
	[
		sav_CXX=$CXX
		sav_CXXFLAGS=$CXXFLAGS
		CXX=$MPICXX
		CXXFLAGS="$CXXFLAGS"
		AC_CHECK_DECL([OPEN_MPI], [mpi_vendor="OpenMPI"],
		[], [#include "mpi.h"])
		AC_CHECK_DECL([MPICH2], [mpi_vendor="MPICH"],
		[], [#include "mpi.h"])
		CXX=$sav_CXX
		CXXFLAGS=$sav_CXXFLAGS
	])

#
# try to set MPI_CXXFLAGS and MPI_LDFLAGS
#
MPI_CXXFLAGS=
MPI_LDFLAGS=
if test "$enable_mpi" = "yes" -a "$mpi_vendor" = "OpenMPI"
then 
     MPI_CXXFLAGS=`$MPICXX --showme:compile`
     MPI_LDFLAGS=`$MPICXX --showme:link`
     AC_MSG_NOTICE([OpenMPI found])
     AC_MSG_NOTICE([MPI_CXXFLAGS=$MPI_CXXFLAGS])
     AC_MSG_NOTICE([MPI_LDFLAGS=$MPI_LDFLAGS])
elif test "$enable_mpi" = "yes" -a "$mpi_vendor" = "MPICH"
then
     # first grab CXXFLAGS (awk script is used to remove CXX at front)
     tmp=`$MPICXX -compile-info | awk '{$1=""; print $0 }'`
     MPI_CXXFLAGS=
     for i in $tmp
     do 
     	 case $i in 
     	      -[[DIUbi]]*)
     		MPI_CXXFLAGS="$MPI_CXXFLAGS $i"
             	;;
     	 esac
     done
     # second grab LDFLAGS (awk script is used to remove CXX at front)
     tmp=`$MPICXX -link-info | awk '{$1=""; print $0 }'`
     for i in $tmp
     do 
     	case $i in 
     	      [[\\/]]*.a | ?:[[\\/]]*.a | -[[lLRu]]* | -Wl* )
     	      	MPI_LDFLAGS="$MPI_LDFLAGS $i"
     	       	;;
     	esac
     done
     AC_MSG_NOTICE([MPICH found])
     AC_MSG_NOTICE([MPI_CXXFLAGS=$MPI_CXXFLAGS])
     AC_MSG_NOTICE([MPI_LDFLAGS=$MPI_LDFLAGS])
else
     AC_MSG_WARN([Neither OpenMPI and MPICH have been recognized...])
fi
AC_SUBST(MPI_CXXFLAGS)
AC_SUBST(MPI_LDFLAGS)


# ------------------------------------------------------------------------------
# Declares output files
# ------------------------------------------------------------------------------
AC_CONFIG_FILES([
	Makefile 
	src/Makefile
	])
AC_OUTPUT

AC_MSG_RESULT([
	--------------------------------------------------
	Configuration summary for $PACKAGE_NAME ($VERSION)
        --------------------------------------------------

	* Installation prefix    : $prefix
	* Build debug            : $DEBUG
	* gprof profiler enabled : $gprof_enabled ($gprof_compiler)
	* timing  report enabled : $timing_enabled
	* Double enabled         : $DOUBLE

	* CUDA tools version     : $NVCC_VERSION
	* MPI enabled            : $enable_mpi (vendor $mpi_vendor)
])

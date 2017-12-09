dnl Process this file with autoconf to produce a configure script.

AC_PREREQ(2.59)
AC_INIT(nls_logsig, 1.0)


AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE()

AC_PROG_CXX
AC_PROG_CC

AC_CHECK_HEADERS([gsl/gsl_cdf.h gsl/gsl_linalg.h],, AC_MSG_ERROR([Cannot find gsl headers]))
AC_CHECK_LIB(gsl, gsl_multifit_fdfsolver_alloc, , AC_MSG_ERROR([Cannot find gsl_multifit_fdfsolver_alloc]), [-lgslcblas])
AC_CHECK_LIB(gsl, gsl_multifit_fdfsolver_free, , AC_MSG_ERROR([Cannot find gsl_multifit_fdfsolver_free]), [-lgslcblas])
AC_CHECK_LIB(gsl, gsl_multifit_fdfsolver_driver, , AC_MSG_ERROR([Cannot find gsl_multifit_fdfsolver_driver]), [-lgslcblas])

# disable default LIBS
AC_SUBST(LIBS, "")

AC_CONFIG_FILES(Makefile src/Makefile)
AC_OUTPUT


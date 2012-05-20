
stopifnot(require(compiler),            # to byte-compile R code
          require(rbenchmark),          # to time our attempts
          require(inline),              # to compile/link/load C++ on the fly
          require(RcppGSL))             # for the GSL example

## Gibbs sampler for function:

## f(x,y) = x x^2 \exp(-xy^2 - y^2 + 2y - 4x)

## using conditional distributions:

## x|y \sim Gamma(3, y^2 +4)
## y|x \sim Normal(\frac{1}{1+x}, \frac{1}{2(1+x)})

## Here is the actual Gibbs Sampler
## This is Darren Wilkinsons R code (with the corrected variance)
## But we are returning only his columns 2 and 3 as the 1:N sequence
## is never used below
Rgibbs <- function(N,thin) {
    mat <- matrix(0,ncol=2,nrow=N)
    x <- 0
    y <- 0
    for (i in 1:N) {
        for (j in 1:thin) {
            x <- rgamma(1,3,y*y+4)
            y <- rnorm(1,1/(x+1),1/sqrt(2*(x+1)))
        }
        mat[i,] <- c(x,y)
    }
    mat
}

## We can also try the R compiler on this R function
RCgibbs <- cmpfun(Rgibbs)



## Now for the Rcpp version -- Notice how easy it is to code up!
gibbscode <- '

  // n and thin are SEXPs which the Rcpp::as function maps to C++ vars
  int N   = as<int>(n);
  int thn = as<int>(thin);

  int i,j;
  NumericMatrix mat(N, 2);

  RNGScope scope;         // Initialize Random number generator

  // The rest of the code follows the R version
  double x=0, y=0;

  for (i=0; i<N; i++) {
    for (j=0; j<thn; j++) {
      x = ::Rf_rgamma(3.0,1.0/(y*y+4));
      y = ::Rf_rnorm(1.0/(x+1),1.0/sqrt(2*x+2));
    }
    mat(i,0) = x;
    mat(i,1) = y;
  }

  return mat;             // Return to R
'

# Compile and Load
RcppGibbs <- cxxfunction(signature(n="int", thin = "int"), gibbscode, plugin="Rcpp")


gslgibbsincl <- '
  #include <gsl/gsl_rng.h>
  #include <gsl/gsl_randist.h>

  using namespace Rcpp;  // just to be explicit
'

gslgibbscode <- '
  int N = as<int>(ns);
  int thin = as<int>(thns);
  int i, j;
  gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
  double x=0, y=0;
  NumericMatrix mat(N, 2);
  for (i=0; i<N; i++) {
    for (j=0; j<thin; j++) {
      x = gsl_ran_gamma(r,3.0,1.0/(y*y+4));
      y = 1.0/(x+1)+gsl_ran_gaussian(r,1.0/sqrt(2*x+2));
    }
    mat(i,0) = x;
    mat(i,1) = y;
  }
  gsl_rng_free(r);

  return mat;           // Return to R
'

## Compile and Load
GSLGibbs <- cxxfunction(signature(ns="int", thns = "int"),
                        body=gslgibbscode, includes=gslgibbsincl,
                        plugin="RcppGSL")

## without RcppGSL, using cfunction()
#GSLGibbs <- cfunction(signature(ns="int", thns = "int"),
#                      body=gslgibbscode, includes=gslgibbsincl,
#                      Rcpp=TRUE,
#                      cppargs="-I/usr/include",
#                      libargs="-lgsl -lgslcblas")

boostincl <- '
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>

typedef boost::mt19937 RNGType; 	// select a generator, MT good default
RNGType rng(123456);			// instantiate and seed

boost::normal_distribution<> n01(0.0, 1.0);
boost::variate_generator< RNGType, boost::normal_distribution<> > rngNormal(rng, n01);

boost::gamma_distribution<> g3(3.0); // older Boost took one arg "alpha", divide draw by "beta"
boost::variate_generator< RNGType, boost::gamma_distribution<> > rngGamma(rng, g3);

'

boostcode <- '
  int N = as<int>(ns);
  int thin = as<int>(thns);
  int i, j;
  double x=0, y=0;
  NumericMatrix mat(N, 2);

  for (i=0; i<N; i++) {
    for (j=0; j<thin; j++) {
      x = rngGamma()/(1.0/(y*y+4));     // dividing by beta gives us Gamma(alpha, beta)
      y = 1.0/(x+1) + rngNormal()*(1.0/sqrt(2*x+2));  // scale by sigma and move by mu
    }
    mat(i,0) = x;
    mat(i,1) = y;
  }
  return mat;           // Return to R
'

BoostGibbs <- cxxfunction(signature(ns="int", thns = "int"),
                          body=boostcode, includes=boostincl,
                          plugin="Rcpp")


cxx12incl <- '#include <random>'

cxx12code <- '
  int N = as<int>(ns);
  int thin = as<int>(thns);
  int i, j;
  double x=0, y=0;
  NumericMatrix mat(N, 2);

  std::mt19937 engine(42);
  std::gamma_distribution<> gamma(3.0, 1.0);
  std::normal_distribution<> normal(0.0, 1.0);

  for (i=0; i<N; i++) {
    for (j=0; j<thin; j++) {
      x = gamma(engine)/(1.0/(y*y+4));     // beta normalized to 1.0; dividing by beta gives us Gamma(alpha, beta)
      y = 1.0/(x+1) + normal(engine)*(1.0/sqrt(2*x+2));  // scale by sigma and move by mu
    }
    mat(i,0) = x;
    mat(i,1) = y;
  }
  return mat;           // Return to R
'

rcppPlugin <- getPlugin("Rcpp")
rcppPlugin$env$PKG_CXXFLAGS <- "-std=c++0x"
Cxx12Gibbs <- cxxfunction(signature(ns="int", thns = "int"),
                          body=cxx12code, includes=cxx12incl,
                          settings=rcppPlugin)

cxxMPincl <- '
#include <random>
#include <omp.h>
'

cxxMPcode <- '
  int N = as<int>(ns);
  int thin = as<int>(thns);
  int i, j;
  double x=0, y=0;
  //NumericMatrix mat(N, 2);
  std::vector<double> X(N), Y(N);

  std::mt19937 engine(42);
  std::gamma_distribution<> gamma(3.0, 1.0);
  std::normal_distribution<> normal(0.0, 1.0);

  omp_set_num_threads(8);
#pragma omp parallel for private(j)

  for (i=0; i<N; i++) {
    for (j=0; j<thin; j++) {
      x = gamma(engine)/(1.0/(y*y+4));     // beta normalized to 1.0; dividing by beta gives us Gamma(alpha, beta)
      y = 1.0/(x+1) + normal(engine)*(1.0/sqrt(2*x+2));  // scale by sigma and move by mu
    }
    X[i] = x;
    Y[i] = y;
  }
  return Rcpp::List::create(Rcpp::Named("x")=X,
                            Rcpp::Named("y")=Y);
'

rcppPlugin <- getPlugin("Rcpp")
rcppPlugin$env$PKG_CXXFLAGS <- "-std=c++0x -fopenmp"
rcppPlugin$env$PKG_LIBS <- paste("-fopenmp", rcppPlugin$env$PKG_LIBS)
CxxMPGibbs <- cxxfunction(signature(ns="int", thns = "int"),
                          body=cxxMPcode, includes=cxxMPincl,
                          settings=rcppPlugin)


## also use rbenchmark package
##
## these values are low as we're still testing, and R is slow on this
N <- 1000
thn <- 100
res <- benchmark(Rgibbs(N, thn),
                 RCgibbs(N, thn),
                 RcppGibbs(N, thn),
                 GSLGibbs(N, thn),
                 BoostGibbs(N, thn),
                 Cxx12Gibbs(N, thn),
                 CxxMPGibbs(N, thn),
                 columns=c("test", "replications", "elapsed",
                           "relative", "user.self", "sys.self"),
                 order="relative",
                 replications=10)
print(res)


## these values are low as we're still testing, and R is slow on this
N <- 2000
thn <- 200
res <- benchmark(RcppGibbs(N, thn),
                 GSLGibbs(N, thn),
                 BoostGibbs(N, thn),
                 Cxx12Gibbs(N, thn),
                 CxxMPGibbs(N, thn),
                 #columns=c("test", "replications", "elapsed",
                 #          "relative", "user.self", "sys.self"),
                 order="relative",
                 replications=20)
print(res)



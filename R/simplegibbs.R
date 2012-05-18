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
require(compiler)
RCgibbs <- cmpfun(Rgibbs)


require(rbenchmark)
## also use rbenchmark package
##
## these values are low as we're still testing, and R is slow on this
N <- 1000
thn <- 100
res <- benchmark(Rgibbs(N, thn),
                 RCgibbs(N, thn),
                 #RcppGibbs(N, thn),
                 #GSLGibbs(N, thn),
                 columns=c("test", "replications", "elapsed",
                           "relative", "user.self", "sys.self"),
                 order="relative",
                 replications=10)
print(res)



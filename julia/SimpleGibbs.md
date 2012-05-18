The Gibbs sampler discussed on [Darren Wilkinson's blog](http://bit.ly/IWhJ52) and also on [Dirk Eddelbuettel's blog](http://dirk.eddelbuettel.com/blog/2011/07/14/) has been implemented in several languages, the first of which was [R](http://www.R-project.org).  

The task is to create a Gibbs sampler for the unscaled density

     f(x,y) = x x^2 \exp(-xy^2 - y^2 + 2y - 4x)

using the conditional distributions

     x|y \sim Gamma(3, y^2 +4)
     y|x \sim Normal(\frac{1}{1+x}, \frac{1}{2(1+x)})

Dirk's version of Darren's original R function is
```r
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
```
Dirk also shows the use of the R byte compiler on this function
```r
RCgibbs <- cmpfun(Rgibbs)
```
In the `examples` directory of the [Rcpp](http://cran.r-project.org/package=Rcpp) package, Dirk provides an `R` script using the [inline](http://cran.r-project.org/package=inline), `Rcpp` and [RcppGSL](http:/cran.r-project.org) packages to implement this sampler in `C++` code callable from `R` and time the results.  On my desktop computer, timing 10 replications of `Rgibbs(20000, 200)` and the other versions produces
```text
               test replications elapsed  relative user.self sys.self
4  GSLGibbs(N, thn)           10   8.228  1.000000     8.224     0.00
3 RcppGibbs(N, thn)           10  13.531  1.644507    13.525     0.00
2   RCgibbs(N, thn)           10 360.652 43.832280   360.198     0.02
1    Rgibbs(N, thn)           10 456.440 55.473991   455.777     0.12
```

A naive translation of Rgibbs to [Julia](http://julialang.org) can use the same samplers for the gamma and normal distributions as does R.  The `C` code for R's d-p-q-r functions for probability densities, cumulative distribution, quantile and random sampling can be compiled into a separate Rmath library.  These sources are included with the Julia sources and Julia functions with similar calling sequences are available as "extras/Rmath.jl"
```julia
load("extras/Rmath.jl")

function JGibbs1(N::Int, thin::Int)
    mat = Array(Float64, (N, 2))
    x   = 0.
    y   = 0.
    for i = 1:N
        for j = 1:thin
            x = rgamma(1,3,(y*y + 4))[1]
            y = rnorm(1, 1/(x+1),1/sqrt(2(x + 1)))[1]
        end
        mat[i,:] = [x,y]
    end
    mat
end
```

You can see that `JGibbs1` is essentially the same code as `Rgibbs` with minor adjustments for syntax.  A similar timing on the same computer gives
```julia
julia> sum([@elapsed JGibbs1(20000, 200) for i=1:10])
27.748079776763916
julia> sum([@elapsed JGibbs1(20000, 200) for i=1:10])
27.782002687454224
```
which is 17 times faster than `Rgibbs` and 13 times faster than `RCgibbs`.  It's actually within a factor of 2 of the compiled code in `RcppGibbs`.

One of the big differences between this function and the compiled C++ function, `RcppGibbs`, is that the compiled function calls the underlying `C` code for the samplers directly, avoiding the overhead of creating a vector of length 1 and indexing to get the first element.  As these operations are done in the inner loop of `JGibbs1` their overhead mounts up.

Fortunately, Julia allows for calling a C function directly.  You need the symbol from the dynamically loaded library, the signature of the function and the arguments.

It looks like
```julia
function JGibbs2(N::Int, thin::Int)
    mat = Array(Float64, (N, 2))
    x   = 0.
    y   = 0.
    for i = 1:N
        for j = 1:thin
            x = ccall(dlsym(_jl_libRmath, :rgamma),
                      Float64, (Float64, Float64), 3., (y*y + 4))
            y = ccall(dlsym(_jl_libRmath, :rnorm),
                      Float64, (Float64, Float64), 1/(x+1), 1/sqrt(2(x + 1)))
        end
        mat[i,:] = [x,y]
    end
    mat
end
```

The timings are considerably faster, essentially the same as `RcppGibbs`
```julia
julia> sum([@elapsed JGibbs2(20000, 200) for i=1:10])
13.596416234970093
julia> sum([@elapsed JGibbs2(20000, 200) for i=1:10])
13.584651470184326
```

If we switch to the native Julia random samplers for the gamma and normal distribution, the function becomes
```julia

function JGibbs3(N::Int, thin::Int)
    mat = Array(Float64, (N, 2))
    x   = 0.
    y   = 0.
    for i = 1:N
        for j = 1:thin
            x = randg(3) * (y*y + 4)
            y = 1/(x + 1) + randn()/sqrt(2(x + 1))
        end
        mat[i,:] = [x,y]
    end
    mat
end
```
and the timings are
```julia
julia> sum([@elapsed JGibbs3(20000, 200) for i=1:10])
6.603794574737549
julia> sum([@elapsed JGibbs3(20000, 200) for i=1:10])
6.58268928527832
```

So now we are beating the compiled code from both `RcppGibbs` (which is using slower samplers) and `GSLGibbs` (faster samplers but not as fast as those in Julia) while writing code that looks very much like the original R function.

**But wait, there's more!**

This computer has a 4-core processor and Julia can take advantage of that.  When starting Julia we specify the number of processes
```bash
julia -p 4
```
and use Julia's tools for parallel execution.  An appealing abstraction in Julia is that of "distributed arrays".  A distributed array is declared like an array with an element type and dimensions plus two additional arguments: the dimension on which to distribute the array to the different processes and a function that states how each section should be constructed.  Usually this is an anonymous function.  We will show two versions of the distributed sampler: the first, `dJGibbs3a`, leaves the result as a distributed array and the second, `dJGibbs3b`, converts the result to a single array in the parent process.

```julia
## Distributed versions - keeping the results as a distributed array
dJGibbs3a(N::Int, thin::Int) = darray((T,d,da)->JGibbs3(d[1],thin), Float64, (N, 2), 1)
## Converting the results to an array controlled by the parent process
dJGibbs3b(N::Int, thin::Int) = convert(Array{Float64,2}, dJGibbs3a(N, thin))
```

Notice that these are one-liners.  In Julia, a function consisting of a single expression can be written by giving the signature and that expression, as shown.  The anonymous function in `dJGibbs3a` is declared with the right-pointing arrow construction `->`.  Its arguments are the type of the array, `T`, the dimensions of the local chunk, `d`, and the dimension on which the array is distributed, `da`.  Here we only use the number of rows, `d[1]`, of the chunk to be generated.

The timings,
```julia
julia> sum([@elapsed dJGibbs3a(20000, 200) for i=1:10])
1.6914057731628418
julia> sum([@elapsed dJGibbs3a(20000, 200) for i=1:10])
1.6724529266357422
julia> sum([@elapsed dJGibbs3b(20000, 200) for i=1:10])
2.2329299449920654
julia> sum([@elapsed dJGibbs3b(20000, 200) for i=1:10])
2.267037868499756
```
are remarkable.  The speed-up with 4 processes leaving the results as a distributed array, which would be the recommended approach if we were going to do further processing, is essentially 4x.  This is because there is almost no communication overhead.  When converting the results to a (non-distributed) array, the speed-up is 3x.

If you haven't looked into Julia before now, you owe it to yourself to do so.

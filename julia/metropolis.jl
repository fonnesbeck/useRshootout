load("extras/Rmath.jl")

# The data
age   = [13., 14, 14,12, 9, 15, 10, 14, 9, 14, 13, 12, 9, 10, 15, 11,
         15, 11, 7, 13, 13, 10, 9, 6, 11, 15, 13, 10, 9, 9, 15, 14, 14,
         10, 14, 11, 13, 14, 10]
price = [2950, 2300, 3900, 2800, 5000, 2999, 3950, 2995, 4500, 2800,
         1990, 3500, 5100, 3900, 2900, 4950, 2000, 3400, 8999, 4000,
         2950, 3250, 3950, 4600, 4500, 1600, 3900, 4200, 6500, 3500,
         2999, 2600, 3250, 2500, 2400, 3990, 4600, 450,4700]/1000.

                        # dnorm log-density method for vector x and mu 
function dnorm(x::Vector{Float64}, mu::Vector{Float64}, sd::Float64, give_log::Bool)
     [dnorm(x[i], mu[i], sd, give_log) for i=1:length(mu)]
end
                                        # log joint posterior density
function calc_posterior(a::Number, b::Number, t::Number,
                        y::Vector{Float64}, x::Vector{Float64})
                                        # Priors on a,b
    logp  = dnorm(a, 0, 10000, true) + dnorm(b, 0, 10000, true)
                                        # Prior on t
    logp += dgamma(t, 0.001, 1000, true)
                                        # Calculate mu
    mu = a + b*x
                                        # Data likelihood
    logp + sum(dnorm(y, mu, 1/sqrt(t), true))
end
                                        # passing a vector [a, b, log(t)]
function calc_posterior(p::Vector{Float64}, y::Vector{Float64}, x::Vector{Float64})
    length(p) == 3 ? calc_posterior(p[1], p[2], exp(p[3]), y, x) :
                     error("argument p must be Array(Float64, (3,))")
end

function sample(iterations::Int, tune_interval::Int,
                y::Vector{Float64}, x::Vector{Float64})
                                        # Initial proposal standard deviations
    n_params = 3
    prop_sd  = fill(5., (n_params,))
                                        # Creat output array trace
    trace    = zeros(Float64, (iterations+1, n_params))
                                        # Initialize acceptance counts
    accepted = zeros(Int, (n_params,))
                                        # log posterior density at initial values
    current_log_prob = calc_posterior(reshape(trace[1,:], (n_params,)), y, x)

    for i=2:(iterations+1)              
        if !bool(i%1000); println("Iteration $i");end

        p = reshape(trace[i-1,:], (n_params,)) # initialize proposal vector
        for j=1:n_params
            p[j] += prop_sd[j] * randn() # proposal for parameter j
                                        # log posterior at proposed vector
            proposed_log_prob = calc_posterior(p, y, x)
                                        # Log-acceptance rate
            alpha = proposed_log_prob - current_log_prob
                                        # Test proposed value
            if log(rand()) < alpha
                current_log_prob = proposed_log_prob # Accept
                accepted[j] += 1
            else
                p[j] = trace[i-1,j]     # revert to previous value
            end
            
            ## Tune the standard deviations every <tune_interval> iterations
            if !bool(i % tune_interval)
                acceptance_rate = accepted[j]/tune_interval
                if acceptance_rate < 0.2
                    prop_sd[j] *= 0.9
                elseif acceptance_rate > 0.5
                    prop_sd[j] *= 1.1
                else
                    println("Parameter $j is tuned!")
                end
                accepted[j] = 0
            end
        end
        trace[i,:] = p
    end
    trace, prop_sd
end


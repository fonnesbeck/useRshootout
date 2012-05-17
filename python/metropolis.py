import numpy as np
import matplotlib as mpl
from scipy.stats import distributions
rnorm = np.random.normal
runif = np.random.rand
dgamma = distributions.gamma.logpdf
dnorm = distributions.norm.logpdf

# The data
age = np.array([13, 14, 14,12, 9, 15, 10, 14, 9, 14, 13, 12, 9, 10, 15, 11, 15, 11, 7, 13, 13, 10, 9, 6, 11, 15, 13, 10, 9, 9, 15, 14, 14, 10, 14, 11, 13, 14, 10])
price = np.array([2950, 2300, 3900, 2800, 5000, 2999, 3950, 2995, 4500, 2800, 1990, 3500, 5100, 3900, 2900, 4950, 2000, 3400, 8999, 4000, 2950, 3250, 3950, 4600, 4500, 1600, 3900, 4200, 6500, 3500, 2999, 2600, 3250, 2500, 2400, 3990, 4600, 450,4700])/1000.

n_params = 3


def calc_posterior(a, b, t, y=price, x=age):
    # Calculate joint posterior, given values for a, b and t

    # Priors on a,b
    logp = dnorm(a, 0, 10000) + dnorm(b, 0, 10000)
    # Prior on t
    logp += dgamma(t, 0.001, 0.001)
    # Calculate mu
    mu = a + b*x
    # Data likelihood
    logp += sum(dnorm(y, mu, t**-2))
    
    return logp


def sample(iterations=10000, tune_interval=100):
        
    # Initial proposal standard deviations
    prop_sd = [5, 5, 5]
    
    # Initialize trace for parameters
    trace = np.empty((iterations+1, n_params))

    # Set initial values
    trace[0] = 0,0,1
    # Initialize acceptance counts
    accepted = [0]*n_params

    # Calculate joint posterior for initial values
    current_log_prob = calc_posterior(*trace[0])
    
    for i in range(iterations):
    
        if not i%1000: print 'Iteration', i
    
        # Grab current parameter values
        current_params = trace[i]
    
        for j in range(n_params):
    
            # Get current value for parameter j
            p = trace[i].copy()
    
            # Propose new value
            if j==2:
                # Ensure tau is positive
                theta = np.exp(rnorm(np.log(current_params[j]), prop_sd[j]))
            else:
                theta = rnorm(current_params[j], prop_sd[j])
            
            # Insert new value 
            p[j] = theta
    
            # Calculate log posterior with proposed value
            proposed_log_prob = calc_posterior(*p)
    
            # Log-acceptance rate
            alpha = proposed_log_prob - current_log_prob
    
            # Sample a uniform random variate
            u = runif()
    
            # Test proposed value
            if np.log(u) < alpha:
                # Accept
                trace[i+1,j] = theta
                current_log_prob = proposed_log_prob
            else:
                # Reject
                trace[i+1,j] = trace[i,j]
    
            # Tune every <tune_interval> iterations
            if not (i+1) % tune_interval:
    
                # Calculate aceptance rate
                acceptance_rate = accepted[j]/tune_interval
    
                if acceptance_rate<0.2:
                    prop_sd[j] *= 0.9
                elif acceptance_rate>0.5:
                    prop_sd[j] *= 1.1
                else:
                    print 'Parameter %i is tuned!' % j
    
                accepted[j] = 0

    return trace

if __name__=='__main__':

    t = sample()
    print np.mean(t, 0)

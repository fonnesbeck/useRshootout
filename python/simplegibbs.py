'''
Gibbs sampler for function:

f(x,y) = x x^2 \exp(-xy^2 - y^2 + 2y - 4x)

using conditional distributions:

x|y \sim Gamma(3, y^2 +4)
y|x \sim Normal(\frac{1}{1+x}, \frac{1}{2(1+x)})
'''
from numpy import zeros, random, sqrt
gamma = random.gamma
normal = random.normal

def gibbs(N=20000, thin=200):
    mat = zeros((N,2))
    x,y = mat[0]
    for i in range(N):
        for j in range(thin):
            x = gamma(3, y**2 + 4)
            y = normal(1./(x+1), 1./sqrt(2*(x+1)))
        mat[i] = x,y

    return mat

import random,math
from time import time

def gibbs(N=50000,thin=1000):
    trace = [[0,0]]*(N/thin)
    x,y = trace[0]
    print "Iter  x  y"
    for i in range(N):
        x=random.gammavariate(3,1.0/(y*y+4))
        y=random.gauss(1.0/(x+1),1.0/math.sqrt(2*x+2))
        if not i%thin: trace[i/thin] = [x,y]

start = time()
gibbs()
print time() - start

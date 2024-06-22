import math
import jax.numpy as jnp
import legendre as lg
from scipy.special import roots_legendre

# Example usage
def vandermonde_test(l=2,d=2):
    print("\nxxxxxxxxx Testing Legendre-Vandermonde matrix xxxxxxxxxxx")
    multi_indices = lg.generate_total_degree_multi_indices(l, d)
    xs = jnp.array([[0.0, 0.0], [0.5, -0.5]])  
    vandermonde_matrix = lg.total_degree_legendre_poly(xs, multi_indices)    
    print(vandermonde_matrix)    
    print("xxxxxxxxxx Done!\n xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

def interpolation_test(xs, xe, l=2,d=1,N=2,f = lambda x: 1./(1+ 25*x**2)):
    print("\nxxxxxxxxx Testing interpolation xxxxxxxxxxxxxxxxxxxxxxxxxx")
    #Get multi indices
    multi_indices = lg.generate_total_degree_multi_indices(l,d)

    #Build the Vandermonde matrix
    V = lg.total_degree_legendre_poly(xs,multi_indices)

    #Evaluate rhs function on that grid
    y = f(xs)

    #Solve a linear system to get polynomial coefficients
    c = jnp.linalg.solve(V,y)

    #Get the Vandermonde evaluation matrix
    Ve = lg.total_degree_legendre_poly(xe,multi_indices)

    #Evaluate the interpolant
    ye_approx = jnp.matmul(Ve,c)

    #Evaluate true rhs
    ye = f(xe)

    # print an error
    print("Relative l2 error", jnp.linalg.norm(ye-ye_approx)/jnp.linalg.norm(ye))
    print("\nxxxxxxxxxx Done!\n xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    

# Total degree approximation test
degree = 75;
dim = 1;
N = math.comb(degree+dim,dim)
Ne = 1000

#Get a grid
xs,_ = roots_legendre(N)
xs = xs.reshape(N,dim)

# Get an evaluation grid        
xe = jnp.array(jnp.linspace(-1,1,Ne)).reshape(Ne,dim)

f = lambda x: 1./(1+ 25*x**2)
interpolation_test(xs,xe, degree,dim,N, f)

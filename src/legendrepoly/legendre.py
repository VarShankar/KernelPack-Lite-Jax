import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from itertools import combinations_with_replacement, permutations, product


@jit
def legendre_poly(n, x):
    P_n_minus_1 = jnp.ones_like(x)
    P_n = x
    def body(k, val):
        P_n_minus_1, P_n = val
        P_n_plus_1 = ((2*k - 1) * x * P_n - (k - 1) * P_n_minus_1) / k
        return P_n, P_n_plus_1
    P_n = jax.lax.fori_loop(2, n+1, body, (P_n_minus_1, P_n))[1]
    return jnp.where(n == 0, P_n_minus_1, jnp.where(n == 1, x, P_n))

def generate_total_degree_multi_indices(l, d):
    indices = set()
    for comb in combinations_with_replacement(range(l + 1), d):
        if sum(comb) <= l:
            permutations_set = set(permutations(comb))
            indices.update(permutations_set)
    
    # Sort indices by total degree and then lexicographically within each total degree
    indices = sorted(indices, key=lambda x: (sum(x), x[::-1]))
    return jnp.array(indices)

def generate_tensor_product_multi_indices(l, d):
    indices = list(product(range(l + 1), repeat=d))
    indices = sorted(set(indices), key=lambda x: (sum(x), x[::-1]))
    return jnp.array(indices)


def generate_hyperbolic_cross_multi_indices(l, d):
    indices = set()
    for comb in product(range(l + 1), repeat=d):
        if comb.count(0) != d and jnp.prod(jnp.array([x for x in comb if x != 0])) <= l:
            indices.add(comb)
    
    indices = sorted(set(indices)
                     , key=lambda x: (sum(x), x[::-1]))
    return jnp.array(indices)



def poly_eval(degrees, some_poly, x):
    legendre_values = [some_poly(degree, x[i]) for i, degree in enumerate(degrees)]
    return jnp.prod(jnp.array(legendre_values), axis=0)

def legendre_poly_eval(xs, multi_indices):
    def vandermonde_single_point(x):
        return vmap(lambda degrees: poly_eval(degrees, legendre_poly, x))(multi_indices)
    
    vandermonde = vmap(vandermonde_single_point)(xs)
    return vandermonde

    
# def gauss_legendre_zeros_1d(n):
#     #Calculate the zeros of the nth degree Legendre polynomial (Gauss-Legendre nodes)."""
#     zeros, _ = roots_legendre(n)
#     return zeros

def tensor_product_grid(ndims, n, univariate_grid_func):
    #Generate tensor-product Legendre zeros for a given number of dimensions and polynomial order."""
    # Get 1D zeros
    zeros_1d = univariate_grid_func(n)[0]
    
    # Create a meshgrid of the zeros for the given number of dimensions
    grids = jnp.meshgrid(*[zeros_1d]*ndims, indexing='ij')
    
    # Reshape the grids to create the tensor-product points
    tensor_product_zeros = jnp.stack(grids, axis=-1).reshape(-1, ndims)
    return tensor_product_zeros    
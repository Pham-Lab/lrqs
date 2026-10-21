"""
DIVOT baseline wrapper for bivariate causal discovery experiments.

This file is based on/adapted from the official implementation released with:
Tu et al. (2022), "Optimal Transport for Causal Discovery"
Source: https://github.com/TURuibo/DIVOT

License: MIT License
Modifications in this repository include integration into our experimental pipeline.
"""
import numpy as np
import jax.numpy as jnp
from jax import value_and_grad, jit, vmap, random

@jit
def loss_jax(params, df_batch, un):
    x, y = df_batch[:, 0], df_batch[:, 1]

    vec = jnp.sort(y) - jnp.sort(params * un)
    return jnp.var(vec)

val_and_grad = value_and_grad(loss_jax)
vmap_val_and_grad_inner = vmap(val_and_grad, in_axes=(None, None, 1), out_axes=0)
vmap_val_and_grad_outer = vmap(vmap_val_and_grad_inner, in_axes=(None, 0, 2), out_axes=0)
vmap_val_and_grad_outer = jit(vmap_val_and_grad_outer)

def get_neighbor_matrix_fixed_num(df_sort, resolution):
    X = df_sort[:, 0]
    n = len(X)
    num_suro_samples = int(n * resolution / 2)
    suro_index = np.arange(-num_suro_samples, num_suro_samples, 1)
    loop = np.arange(num_suro_samples, n - num_suro_samples, 1)
    
    neighbor_matrix = np.zeros([n, n], dtype=bool)
    for j in loop:
        suro_index_pos_j = suro_index + j
        fill_in_false = np.zeros(n, dtype=bool)
        fill_in_false[suro_index_pos_j] = True
        neighbor_matrix[:, j] = fill_in_false
        neighbor_matrix[j, j] = False
    return neighbor_matrix

def get_batches(data, neighborM, resolution, npos):
    n, _ = np.shape(data)
    id_batchLoop = np.arange(0, n, int(n / npos))
    batches = []
    for id_batch in id_batchLoop:
        batch_indx = neighborM[:, id_batch]
        batch = data[batch_indx, :]
        if len(batch) > 0:
            batches.append(batch)
    return batches

def batch_test(df, resolution, npos):
    nghM = get_neighbor_matrix_fixed_num(df, resolution)
    batches = get_batches(data=df, neighborM=nghM, resolution=resolution, npos=npos)
    return jnp.array(batches)

def test(batches, key, step_sz=1.0, exp=100, nrep=50):
    df_batch = batches[0]
    batch_sz, _ = df_batch.shape
    params = 0.2  

    loss_res = []
    for j in range(exp):
        key, subkey = random.split(key)
        un = random.uniform(subkey, shape=(batch_sz, nrep, len(batches)), minval=0.0, maxval=1.0)
        loss_val, grad = vmap_val_and_grad_outer(params, batches, un)
        ave_loss, ave_grad = np.mean(loss_val), np.mean(grad)
        params -= step_sz * ave_grad
        loss_res.append(ave_loss)
        
    return loss_res, params

def get_loss(loss_res, params):
    return np.mean(loss_res[-10:]) / (params**2)

def predict_causal_direction_divot(x, y, seed=42):

    resolution = 0.1
    npos = 10

    df_c = np.column_stack((x, y))
    df_sort_c = df_c[np.argsort(df_c[:, 0])]
    c_batches = batch_test(df_sort_c, resolution, npos)
    
    key = random.PRNGKey(seed)
    c_loss_res, params_c = test(c_batches, key)
    dir_c_loss = get_loss(c_loss_res, params_c)

    df_rv = np.column_stack((y, x))
    df_sort_rv = df_rv[np.argsort(df_rv[:, 0])]
    rv_batches = batch_test(df_sort_rv, resolution, npos)
    
    key = random.PRNGKey(seed)
    rv_loss_res, params_rv = test(rv_batches, key)
    dir_rv_loss = get_loss(rv_loss_res, params_rv)

    if dir_c_loss < dir_rv_loss:
        return 1
    else:
        return -1
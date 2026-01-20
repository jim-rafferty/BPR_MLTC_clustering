
import numpyro
import numpyro.distributions as dist
import jax
import arviz as az
import numpy as np
import optax
import tqdm
import jax.numpy as jnp


from numpyro.contrib.einstein import MixtureGuidePredictive, SteinVI, SVGD, RBFKernel

import gc
from contextlib import nullcontext
# import funsor



def sample_lrvb_posterior(rng_key, lrvb_mean, lrvb_cov, loc_samples, guide, sample_shape=()):
    """
    Sample from the LRVB-corrected multivariate normal posterior.

    Args:
        rng_key: a JAX PRNGKey
        lrvb_mean: [D] mean vector (unconstrained)
        lrvb_cov: [D, D] covariance matrix (unconstrained)
        guide: The variatioßnal distribution function
        sample_shape: shape of the sample batch (e.g., (1000,) for 1000 samples)

    Returns:
        samples_dict: a dict mapping latent variable names to constrained samples
                      of shape `sample_shape + latent_shape`
    """
    unconstrained_samples = dist.MultivariateNormal(
        loc=lrvb_mean, covariance_matrix=lrvb_cov
    ).sample(rng_key, sample_shape)

    latent_names = list(loc_samples.keys())
    latent_names_clean = [n.replace("_auto_loc", "") for n in latent_names]

    # Need to remember what the unconstrained shapes are so we can transform back
    unconstrained_shapes = [loc_samples[name].shape for name in latent_names]

    sizes = [int(np.prod(shape)) for shape in unconstrained_shapes]
    split_indices = np.cumsum(np.array(sizes[:-1]))

    unconstrained_samples_flat = unconstrained_samples.reshape((-1, lrvb_mean.shape[0]))
    split_unconstrained = np.split(unconstrained_samples_flat, split_indices, axis=-1)

    # Apply projections from guide to correctly constrain parameters 
    samples_dict = {}
    for name, raw_sample, shape in zip(latent_names_clean, split_unconstrained, unconstrained_shapes):
        reshaped = raw_sample.reshape(sample_shape + shape)
        samples_dict[name] = reshaped

    return guide._constrain(samples_dict)


def combine_hessian_with_meanfield(H_subset, subset_keys, opt_params, loc_keys, loc_arrays):
    """
    Combine subset Hessian with mean field diagonal for excluded parameters.
    
    Args:
        H_subset: Subset Hessian matrix 
        subset_keys: Keys of parameters in H_subset
        opt_params: Full parameter dict
        loc_keys: All location parameter keys
        loc_arrays: All location parameter arrays
    
    Returns:
        H_full: Complete Hessian matrix
    """

    # Mean field diagonal: 1 / scale^2
    scale_keys = [k.replace("_loc", "_scale") for k in loc_keys]
    scale_arrays = [opt_params[k] for k in scale_keys]
    flat_scales = np.concatenate([arr.flatten() for arr in scale_arrays])
    meanfield_diag = 1.0 / (flat_scales ** 2)
    
    # Create full Hessian
    H_full = np.diag(meanfield_diag)  # Start with mean field diagonal


    # Find which indices belong to subset

    if isinstance(subset_keys[0], list):
        loop_lim = len(subset_keys)
    else:
        loop_lim = 1
        subset_keys = [subset_keys]
        H_subset = [H_subset]

    for idx in range(loop_lim):

        subset_indices = []
        current_idx = 0
        for key, arr in zip(loc_keys, loc_arrays):
            if key in subset_keys[idx]:
                subset_indices.extend(range(current_idx, current_idx + arr.size))
            current_idx += arr.size
        
        subset_indices = np.array(subset_indices)
        
        # Insert subset Hessian in correct positions
        H_full[np.ix_(subset_indices, subset_indices)] = H_subset[idx]
    
    return H_full

def hessian_cookbook(fun):
    # Implementation from JAX's autodiff cookbook
    # call like this:
    #hess_fn = hessian(elbo_loss_fn)
    #H = hess_fn(flat_loc_concat)
    return jax.jit(
        jax.jacfwd(
            jax.jacrev(
                fun
            )
        )
    )


def hessian(f, x):
    """Re-linearize for each hvp to save memory"""
    grad_f = jax.grad(f)
    
    @jax.jit
    def hvp_fn(v):
        _, hvp = jax.linearize(grad_f, x)
        return hvp(v)

    basis = np.eye(np.prod(x.shape)).reshape(-1, *x.shape)
    
    #if verbose:
    #    list_res = [hvp_fn(e) for e in tqdm.tqdm(basis, desc=f"Computing Sub-Hessian {idx}")]
    #else:
    list_res = [hvp_fn(e) for e in basis]
    
    return np.stack(list_res).reshape(x.shape + x.shape)


def hessian_fd(f, x, step_size, verbose):

    n = len(x)
    H = jnp.zeros((n, n))
    f_jit = jax.jit(f)
    
    #TODO - idea, average some of these together?
    step_sizes = jnp.maximum(jnp.abs(x), 1.0) * step_size
    f_x = f_jit(x) 

    with tqdm.tqdm(total = n * (n+1) // 2, desc="Computing Hessian..." ) if verbose else nullcontext() as pbar:
        for i in range(n):
            for j in range(i, n):  # j >= i

                # Compute second-order finite difference
                if i == j:
                    
                    ei = jnp.zeros(n)
                    ei = ei.at[i].set(step_sizes[i])
                    ej = jnp.zeros(n)
                    ej = ej.at[j].set(step_sizes[j])

                    val = (f_jit(x + ei) - 2 * f_x + f_jit(x - ei)) / (step_sizes[i] ** 2)
                    H = H.at[i, j].set(val)
                else:
                    
                    e_pp = jnp.zeros(n)
                    e_pm = jnp.zeros(n)

                    e_pp = e_pp.at[i].set(step_sizes[i] / 2)
                    e_pp = e_pp.at[j].set(step_sizes[j] / 2)

                    e_pm = e_pm.at[i].set(step_sizes[i] / 2)
                    e_pm = e_pm.at[j].set(-step_sizes[j] / 2) 

                    e_mp = -e_pm
                    e_mm = -e_pp

                    val = (f_jit(x + e_pp) - f_jit(x + e_pm) - f_jit(x + e_mp) + f_jit(x + e_mm)) / (step_sizes[i] * step_sizes[j])
                    H = H.at[i, j].set(val)  
                    H = H.at[j, i].set(val)
                if verbose:  
                    pbar.update(1)

    eig_vals, eig_vecs = np.linalg.eigh(H) 

    if verbose and any(eig_vals < 0):
        print(f"Applying regularisation")
        regularisation_value = -eig_vals.min()
        print((eig_vals < 0).sum())
        print(regularisation_value)
        eig_vals += regularisation_value + 1e-4
        print(np.sum(eig_vals < 0))
        print(eig_vals.min())
        H = eig_vecs @ np.diag(eig_vals) @ np.linalg.inv(eig_vecs)

    return H




def NUTS(
    model,
    warmup,
    samples,
    chains,
    discrete_latent=[],
    verbose=True,
    **kwargs
):
    
    kernel = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(
        kernel, 
        num_warmup=warmup, 
        num_samples=samples, 
        num_chains=chains,
        progress_bar=verbose,
    )
    
    rng = jax.random.PRNGKey(5) # TODO - optional arguments
    mcmc.run(
        rng,
        **kwargs
    )

    
    posterior_samples = mcmc.get_samples()
    posterior_predictive_function = numpyro.infer.Predictive(model, posterior_samples, infer_discrete=True)
    posterior_predictive_samples = posterior_predictive_function(rng, **kwargs)
    
    if len(discrete_latent) > 0:
        chain_discrete_samples = jax.tree.map(
            lambda x: x.reshape((chains, samples) + x.shape[1:]),
            {k:posterior_predictive_samples[k] for k in discrete_latent})
        mcmc.get_samples().update(posterior_predictive_samples)
        mcmc.get_samples(group_by_chain=True).update(chain_discrete_samples)
    
    az_obj = az.from_numpyro(
        posterior=mcmc,
        posterior_predictive=posterior_predictive_samples
    )
    return az_obj

class SVIResult:
    def __init__(self, params, losses):
        self.params = params
        self.losses = losses

def SVI(
    model,
    steps,
    samples,
    guide=None,
    elbo_fn=None,
    learning_rate=0.01,
    particles=1,
    discrete_latent=[],
    verbose=True,
    log_lik_compute=False,
    **kwargs
):

    rng = jax.random.PRNGKey(0)

    if guide is None:
        if len(discrete_latent) == 0:
            guide = numpyro.infer.autoguide.AutoMultivariateNormal(model)
        else:
            guide = numpyro.infer.autoguide.AutoMultivariateNormal(
                numpyro.handlers.block(
                    numpyro.handlers.seed(model, rng),
                    hide=discrete_latent
                )
            )
            
    else:
        discrete_latent = [] # If we are not using an autoguide we don't need to mask any variables
        # NOTE - this may not be true if passing in an autoguide.


    if elbo_fn is None:
        if len(discrete_latent) == 0:
            elbo = numpyro.infer.Trace_ELBO(num_particles=particles, vectorize_particles=True) 
        else: 
            elbo = numpyro.infer.TraceEnum_ELBO(num_particles=particles, vectorize_particles=True) 
    else:
        elbo = elbo_fn(num_particles=particles, vectorize_particles=True)
   
   
    if hasattr(learning_rate, "__len__"):

        # piecewise constant learning rate halving each time
        scale = 0.5
        jumps = int(np.log2(learning_rate[0]/learning_rate[1]))
        boundaries = {
            int(i):scale for i in np.linspace(0, steps, jumps + 3)
        }
        _ = boundaries.pop(0)

        schedule = optax.piecewise_constant_schedule(
            init_value=learning_rate[0],
            boundaries_and_scales=boundaries
        )
        optimiser = optax.adam(schedule)
    else:
        # Use a constant learning rate
        optimiser = optax.adam(learning_rate)
        

    svi = numpyro.infer.SVI(
        model, 
        guide, 
        optimiser,
        elbo, 
        **kwargs
    )
    svi_result = svi.run(rng, steps, progress_bar=verbose) 

    ##################################################
    # Manual SVI loop here
    #svi = numpyro.infer.SVI(
    #    model, 
    #    guide, 
    #    optimiser,
    #    elbo,
    #)
    #
    #svi_state = svi.init(
    #    rng, 
    #    **kwargs
    #)
    #
    #losses = np.zeros(steps)
    #
    #def SVI_fit_fn(idx, carry):
    #    svi_state, losses = carry
    #    svi_state, loss = svi.update(svi_state, **kwargs)
    #    losses = losses.at[idx].set(loss)
    #    return (svi_state, losses)
    #
    #if verbose:
    #    SVI_fit_fn = loop_tqdm(steps, message="")(SVI_fit_fn) # TODO - add a more informative message
    #
    #svi_state, losses = jax.lax.fori_loop(0, steps, SVI_fit_fn, (svi_state, losses))
    #
    #svi_result = SVIResult(
    #    svi.get_params(svi_state),
    #    losses
    #)

    ##################################################

    log_lik = {}
    posterior_predictive_samples = {}

    params = svi_result.params

    # display summary of quadratic approximation
    posterior_samples = guide.sample_posterior(
        rng, 
        params,
        sample_shape=(samples,)
    )

    if log_lik_compute: 
        # TODO probably some fixes needed here.

        # calculate the posterior predictive variables first
        posterior_predictive_function = numpyro.infer.Predictive(
            model, 
            guide=guide, 
            params=svi_result.params, 
            num_samples=samples
        )
        
        _, rng_subkey = jax.random.split(key=rng)

        if "N_batch" in kwargs:
            # log likelihood doesn't support batching so we need to do it manually
            orig_n_batch = kwargs["N_batch"]
            kwargs["N_batch"] = None
    
        posterior_predictive_samples = posterior_predictive_function(
            rng_subkey, 
            **kwargs
        )     

        # Add parameters to the posterior samples from the posterior predictive
        for k in set(posterior_predictive_samples.keys()).union(discrete_latent):
            if not(k.startswith("obs")) and k not in posterior_samples:
                posterior_samples[k] = posterior_predictive_samples[k]


        log_lik = numpyro.infer.util.log_likelihood(
            model, 
            posterior_samples,
            **kwargs
        )

        if "N_batch" in kwargs:
            kwargs["N_batch"] = orig_n_batch


    az_obj = az.from_dict(
        posterior={k: v.reshape([1] + list(v.shape)) for k, v in posterior_samples.items()},
        posterior_predictive={k: v.reshape([1] + list(v.shape)) for k, v in posterior_predictive_samples.items()},
        log_likelihood={k: v.reshape([1] + list(v.shape)) for k, v in log_lik.items()},
        sample_stats={"losses": svi_result.losses} # Note, this is not ideal...
    )

    return az_obj

def SteinVI(
    model,
    steps,
    samples,
    learning_rate=0.05,
    particles=1,
    verbose=True,
    log_lik_compute=False,
    **kwargs
):
    
    rng = jax.random.PRNGKey(0)       
    optimiser = numpyro.optim.Adagrad(learning_rate)
    kernel = RBFKernel()

    stein = SVGD(
        model, 
        optimiser, 
        kernel,
        num_stein_particles=particles
    )

    stein_result = stein.run(
        rng, 
        steps, 
        progress_bar=verbose, 
        **kwargs
    ) 

    params = stein_result.params
    for k in params:
        print("params", k, params[k].shape)

    #log_lik = {}
    #posterior_predictive_samples = {}

    posterior_function = numpyro.infer.Predictive(
        stein.guide,
        params=params,
        num_samples=samples,
        batch_ndims=1
    )
    posterior_samples = posterior_function(
        rng,
        **kwargs
    )

    for k in posterior_samples:
        print(k, posterior_samples[k].shape)

    az_obj = az.from_dict(
        posterior={k: v.reshape([1] + list(v.shape)) for k, v in posterior_samples.items()},
    )

    return az_obj
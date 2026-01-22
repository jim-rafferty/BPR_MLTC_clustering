
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

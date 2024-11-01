

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.funsor import config_enumerate

import jax
import jax.numpy as jnp
import numpy as np

import arviz as az

import joblib
from tqdm import tqdm

from functools import partial 

import contextlib


#######################################
# General functions and classes

def mix_weights(beta):
    """
    Function to do the stick breaking construction
    """
    beta1m_cumprod = jnp.cumprod(1 - beta, axis=-1)
    term1 = jnp.pad(beta, (0, 1), mode='constant', constant_values=1.)
    term2 = jnp.pad(beta1m_cumprod, (1, 0), mode='constant', constant_values=1.)
    return jnp.multiply(term1, term2)



def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


class CategoricalUInt8(dist.CategoricalProbs):
    """
    This is a subclassed categorical distribution that changes the sampling 
    data type output to UInt8 to save memory.
    """
    def __init__(self, probs, validate_args=None):
        super().__init__(probs=probs, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        assert numpyro.util.is_prng_key(key)
        return dist.util.categorical(key, self.probs, shape=sample_shape + self.batch_shape).astype(jnp.uint8) 
        
#######################################
# Profile regression model

def profile_regression_model(
    K, 
    N, 
    D_discrete, 
    D_response,
    X_response=None,
    X_discrete=None,
    y=None,
    alpha=None,
    cluster_gradients=False
):  
    if y is not None:
        y = jnp.array(y)
        
    # priors
    if alpha is None:
        alpha = numpyro.sample("alpha", dist.Uniform(0.3, 10.0))
    
    with numpyro.plate("v_plate", K-1):
        v = numpyro.sample("v", dist.Beta(1, alpha))
        
    # Phis (the probabilities in the discrete mixture model)
    with numpyro.plate('discrete_components', D_discrete):
        with numpyro.plate("discrete_cluster", K):
            phi = numpyro.sample('phi', dist.Uniform(0.0, 1.0))

    # priors for the response model. Set to be the same as the premium package dafaults
    with numpyro.plate("response_components", D_response):
        
        with contextlib.ExitStack() as stack:
            if cluster_gradients:
                stack.enter_context(numpyro.plate("response_cluster_gradient", K))
                
            beta = numpyro.sample('beta', dist.StudentT(df=7.0, loc=0.0, scale=2.5))
    
    with numpyro.plate("response_cluster", K):
        intercepts = numpyro.sample("intercepts", dist.StudentT(df=7.0, loc=0.0, scale=2.5))
        
    # model sampling
    with numpyro.plate('data', N):
        
        # Assignment is which cluster each row of data belongs to.
        assignment =  numpyro.sample(
            'assignment',
            CategoricalUInt8(mix_weights(v))
        )

        
        # 1) Discrete mixture model
        # At the moment this assumes independent probabilities
        obs1 = numpyro.sample(
            'obs1', 
            #MultivariateBernoulli(phi[assignment, :]), 
            dist.Bernoulli(phi[assignment, :]).to_event(1),
            obs=X_discrete if X_discrete is not None else None,
        )
        
        # 2) Response model (logistic regression model)
        if cluster_gradients:
            linear_predictor = intercepts[assignment] + jnp.sum(beta[assignment, :] * X_response, axis=-1)
        else:
            linear_predictor = intercepts[assignment] + jnp.sum(beta * X_response, axis=-1)
        obs2 = numpyro.sample(
            "obs2",
            dist.Bernoulli(logits=linear_predictor),
            obs=y if y is not None else None
        )            


#@partial(
#    jax.jit, 
#    static_argnames=[
#        "K", "n_iters", "samples", "cluster_gradients", "posterior_predictive", "return_params"
#    ]
#)
def _fit_profile_regression_model_SVI_jit(
    X_discrete,
    X_response,
    outcome,
    K,
    rng_key,
    n_iters,
    samples,
    alpha=None,
    cluster_gradients=False,
    posterior_predictive=False,
    return_PPDF=False
):

    if return_PPDF and not(posterior_predictive):
        raise Exception("posterior_predictive must be true if return_PPDF is true")
    
    N, D_d = X_discrete.shape
    _, D_r = X_response.shape
    
    guide = numpyro.infer.autoguide.AutoNormal(
        numpyro.handlers.block(
            numpyro.handlers.seed(config_enumerate(profile_regression_model), rng_key),
            hide_fn=lambda site: site["name"] in ["assignment"]
        )
    )
    
    svi = numpyro.infer.svi.SVI(
        config_enumerate(profile_regression_model),
        guide,
        numpyro.optim.Adam(step_size=0.005),
        numpyro.infer.TraceEnum_ELBO(),
        K=K,
        N=N,
        D_discrete=D_d,
        D_response=D_r,
        X_response=X_response,
        X_discrete=X_discrete,
        y=outcome,
        alpha=alpha,
        cluster_gradients=cluster_gradients
    )
    
    init_state = svi.init(rng_key)
    state, loss = jax.lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(n_iters))  
    params = svi.get_params(state)

    posterior_predictive_samples = None
    if posterior_predictive:
        posterior_predictive_function = numpyro.infer.Predictive(
            config_enumerate(profile_regression_model), 
            guide=guide, 
            params=params, 
            num_samples=samples
        )
        
        rng_key, rng_subkey = jax.random.split(key=rng_key)
    
        posterior_predictive_samples = posterior_predictive_function(
            rng_subkey, 
            K=K,
            N=N,
            D_discrete=D_d,
            D_response=D_r,
            X_response=X_response,
            alpha=alpha,
            cluster_gradients=cluster_gradients
        )

    posterior_function = numpyro.infer.Predictive(
        guide,
        params=params,
        num_samples=samples
    )
    posterior_samples = posterior_function(
        rng_key,
        K=K,
        N=samples,
        D_discrete=D_d,
        D_response=D_r,
        X_response=X_response,
        alpha=alpha,
        cluster_gradients=cluster_gradients
    )

    
    log_lik_out = None
    if posterior_predictive:
        posterior_samples["assignment"] = posterior_predictive_samples["assignment"]
    
        
        log_lik = numpyro.infer.util.log_likelihood(
            config_enumerate(profile_regression_model), 
            posterior_samples,
            N=N,
            K=K,
            D_discrete=D_d,
            D_response=D_r,
            X_discrete=X_discrete, 
            X_response=X_response, 
            y=outcome,
            alpha=alpha,
            cluster_gradients=cluster_gradients
        )

        log_lik_out = log_lik["obs1"] + log_lik["obs2"]

    if return_PPDF:
        return posterior_samples, posterior_predictive_samples, log_lik_out, loss, posterior_predictive_function    
    return posterior_samples, posterior_predictive_samples, log_lik_out, loss


def fit_profile_regression_model_MCMC(
    X_discrete,
    X_response,
    outcome,
    K,
    warmup,
    samples,
    alpha,
    cluster_gradients=False
):

    """
    Fits a profile regression model using MCMC and returns an arviz data 
    object. 
    
    Parameters
    -----------------------------
        X_discrete: numpy.array
            The discrete data for use in fitting the model. This data
            should be N * D_d dimensional, where N is the number of individuals
            and D_r is the number of variables.  
        
        X_response: numpy.array
            The data to use to fit the response sector of the model. This data
            should be N * D_r dimensional, where N is the number of individuals
            and D_r is the number of variables. Note, D_d does not have to be 
            equal to D_r
            
        outcome: numpy.array
            The outcome variable to use to fit the response sector of the model.
            This data should be N * 1 dimensional.
            
        K: int
            The number of clusters.
            
        warmup: int
            The number of warmup samples to compute 
            
        samples: int
            The number of posterior samples to compute
            
        alpha: float
            The alpha parameter in the Dirichlet process (set to None to fit this 
            parameter as part of the model fitting)    

        cluster_gradients: bool (default False)
            Whether or not to fit gradients for each of the predictor in each of the 
            identified clusters (True) or global gradients (False). An intercept for each
            cluster is always fit.            
    """
    
    rng_key = jax.random.PRNGKey(0)

    N, D_d = X_discrete.shape
    _, D_r = X_response.shape

    kernel = numpyro.infer.NUTS(profile_regression_model)
    kernel2 = numpyro.infer.DiscreteHMCGibbs(kernel)

    mcmc = numpyro.infer.MCMC(
        kernel2,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=1,
        progress_bar=False
    )
    mcmc.run(
        rng_key,
        K=K,
        N=N, 
        D_discrete=D_d, 
        D_response=D_r,
        X_discrete=X_discrete,
        X_response=X_response,
        y=outcome,
        alpha=alpha,
        cluster_gradients=cluster_gradients
    ) 
    posterior_samples = mcmc.get_samples()
    posterior_predictive_function = numpyro.infer.Predictive(profile_regression_model, posterior_samples, infer_discrete=True)
    posterior_predictive_samples = posterior_predictive_function(
        rng_key,
        K=K,
        N=N, 
        D_discrete=D_d, 
        D_response=D_r,
        X_discrete=X_discrete,
        X_response=X_response,
        y=outcome,
        alpha=alpha,
        cluster_gradients=cluster_gradients
    )

    log_lik = numpyro.infer.util.log_likelihood(
        profile_regression_model, posterior_samples,
        K=K,
        N=N, 
        D_discrete=D_d, 
        D_response=D_r,
        X_discrete=X_discrete,
        X_response=X_response,
        y=outcome,
        alpha=alpha,
        cluster_gradients=cluster_gradients
    )

    inf_dict = {k: v.reshape( [1] + list(v.shape)) for k, v in posterior_samples.items() if k != "assignment"}
    posterior_predictive_dict = {k: v.reshape( [1] + list(v.shape)) for k, v in posterior_predictive_samples.items()}

    out = az.from_dict(
        inf_dict, 
        posterior_predictive=posterior_predictive_dict,
        observed_data={"obs1":X_discrete, "obs2":outcome},
        log_likelihood={"obs":(log_lik["obs1"] + log_lik["obs2"]).reshape([1] + list(log_lik["obs1"].shape))}
    )

    return out


def fit_profile_regression_model(
    X_discrete,
    X_response,
    outcome,
    K,
    n_iters,
    samples,
    alpha,
    cluster_gradients=False,
    posterior_predictive=False
):
    
    """
    Fits a profile regression model using SVI and returns an arviz data 
    object and an array recording the loss at each step. 
    
    Parameters
    -----------------------------
        X_discrete: numpy.array
            The discrete data for use in fitting the model. This data
            should be N * D_d dimensional, where N is the number of individuals
            and D_r is the number of variables.  
        
        X_response: numpy.array
            The data to use to fit the response sector of the model. This data
            should be N * D_r dimensional, where N is the number of individuals
            and D_r is the number of variables. Note, D_d does not have to be 
            equal to D_r
            
        outcome: numpy.array
            The outcome variable to use to fit the response sector of the model.
            This data should be N * 1 dimensional.
            
        K: int
            The number of clusters.
            
        alpha: float
            The alpha parameter in the Dirichlet process (set to None to fit this 
            parameter as part of the model fitting)    
            
        n_iters: int
            The number of iterations to use in the SVI fitting
            
        samples: int
            The number of posterior samples to compute

        cluster_gradients: bool (default False)
            Whether or not to fit gradients for each of the predictor in each of the 
            identified clusters (True) or global gradients (False). An intercept for each
            cluster is always fit.
            
        posterior_predictive: bool (default False)
            Whether or not to compute the posterior predictive distribution (ie whether 
            to return samples of the latent variables in the model). This can use quite
            a lot of memory, especially when the dataset is large.
        
    """
    
    rng_key = jax.random.PRNGKey(0)
    
    N, D_d = X_discrete.shape
    _, D_r = X_response.shape
    
    posterior_samples, posterior_predictive_samples, log_lik_in_sample, loss = _fit_profile_regression_model_SVI_jit(
        X_discrete,
        X_response,
        outcome,
        K,
        rng_key,
        n_iters,
        samples,
        alpha=alpha,
        cluster_gradients=cluster_gradients,
        posterior_predictive=posterior_predictive
    )

    if posterior_predictive:
        posterior_samples["assignment"] = posterior_predictive_samples["assignment"]
        log_lik = numpyro.infer.util.log_likelihood(
            config_enumerate(profile_regression_model), 
            posterior_samples,
            K=K,
            N=N, 
            D_discrete=D_d,
            D_response=D_r,
            X_discrete=X_discrete,
            X_response=X_response,
            y=outcome,
            alpha=alpha,
            cluster_gradients=cluster_gradients
        )
        
        posterior_predictive_dict = {
            k: v.reshape([1] + list(v.shape)) for k, v in posterior_predictive_samples.items()
        }
        
    inf_dict = {k: v.reshape( [1] + list(v.shape)) for k, v in posterior_samples.items() if k != "assignment"}
    
    if posterior_predictive:
        out = az.from_dict(
            inf_dict, 
            posterior_predictive=posterior_predictive_dict,
            observed_data={"obs1":X_discrete},
            log_likelihood={"obs":(log_lik["obs1"] + log_lik["obs2"]).reshape([1] + list(log_lik["obs1"].shape))}
        )
    else:
        out = az.from_dict(
            inf_dict, 
            observed_data={"obs1":X_discrete}
        )
    
    return out, loss


# ELPD tools:

def _k_fold_ELPD_PR_single_thread(
    X_discrete,
    X_response,
    y,
    K,
    D_d,
    D_r,
    i,
    rng_key,
    n_iters,
    samples,
    alpha,
    cluster_gradients
):

    posterior_samples, posterior_predictive_samples, log_lik_in_sample, loss, posterior_predictive_function = _fit_profile_regression_model_SVI_jit(
        np.delete(X_discrete, i, axis=0),
        np.delete(X_response, i, axis=0),
        np.delete(y, i, axis=0),
        K,
        rng_key,
        n_iters,
        samples,
        alpha,
        cluster_gradients,
        posterior_predictive=True,
        return_PPDF=True
    )
    
    # generate posterior predictive samples for the assignment of the unobserved measurements
    rng_key, rng_subkey = jax.random.split(key=rng_key)

    # NOTE(Jim): This PPD function was passed out of the model fitting function above.
    # I tried but I couldn't construct this PPD function here so that it works. The
    # error looked like:
    # ValueError: Continuous inference cannot handle discrete sample site 'obs1'.
    # Passing this function out means we can't jit compile the _fit_profile_regression... function
    # as jax doesn't support function types.  
    pps = posterior_predictive_function(
        rng_subkey, 
        K=K,
        N=len(i),
        D_discrete=D_d,
        D_response=D_r,
        X_response=X_response[i, :],
        alpha=alpha,
        cluster_gradients=cluster_gradients
    )

    
    posterior_samples["assignment"] = pps["assignment"]
    
    log_lik = numpyro.infer.util.log_likelihood(
        config_enumerate(profile_regression_model), 
        posterior_samples,
        N=len(i), 
        K=K,
        D_discrete=D_d,
        D_response=D_r,
        X_discrete=X_discrete[i, :],
        X_response=X_response[i, :],
        y=y[i],
        alpha=alpha,
        cluster_gradients=cluster_gradients
    )

    return log_lik["obs1"] + log_lik["obs2"], log_lik_in_sample

def k_fold_ELPD_profile_regression(
    X_discrete,
    X_response,
    outcome,
    K, 
    n_iters,
    samples,
    folds=None,
    parallel=-1,
    verbose=False,
    alpha=0.1,
    cluster_gradients=False,
    ppd_sampling_factor=1.0
):
    """
    Estimates the ELPD of a profile regression model using cross validation. The 
    model is fit using SVI. Returns the ELPD estimate.
    
    Parameters
    -----------------------------
        X_discrete: numpy.array
            The discrete data for use in fitting the model. This data
            should be N * D_d dimensional, where N is the number of individuals
            and D_r is the number of variables.  
        
        X_response: numpy.array
            The data to use to fit the response sector of the model. This data
            should be N * D_r dimensional, where N is the number of individuals
            and D_r is the number of variables. Note, D_d does not have to be 
            equal to D_r
            
        outcome: numpy.array
            The outcome variable to use to fit the response sector of the model.
            This data should be N * 1 dimensional.
            
        K: int
            The number of clusters.

        alpha: float
            The alpha parameter in the Dirichlet process (set to None to fit this 
            parameter as part of the model fitting)
            
        n_iters: int
            The number of iterations to use in the SVI fitting
            
        samples: int
            The number of posterior samples to compute
            
        folds: int, (default N)
            The number of folds to divide the data into for cross validation. This 
            is equal to the number of times the model must be fitted.
            
        parallel: int, (default -1 - all processors are used)
            Number of processors accross which to split the fitting of the folds.
            Performance may not improve using parallel processing if the data and the
            number of folds are both relatively small. 

        verbose: bool (default False)
            Whether or not to print a short string with details of the ELPD results 
            following computation
            
        cluster_gradients: bool (default False)
            Whether or not to fit gradients for each of the predictor in each of the 
            identified clusters (True) or global gradients (False). An intercept for each
            cluster is always fit.
            
        ppd_sampling_factor: float 0 < x <= 1.0 (default 1.0)
            The fraction of the dataset to use to calculate the posterior predictive 
            distribution. Setting this to a number smaller than 1.0 means the dataset
            is randomly subsampled, and saves memory.
        
    """
    
    N, D_d = X_discrete.shape
    _, D_r = X_response.shape
    
    if folds is None:
        folds = N
    
    rng_key = jax.random.PRNGKey(0)

    N_ppd = int(ppd_sampling_factor * N)
    if N_ppd > N or N_ppd <= 0:
        raise Exception("ppd_sampling_factor must be less than or equal to 1 and greater than 0. Requested value = {}".format(ppd_sampling_factor))


    if N_ppd < N:
        idx = np.random.choice(N, N_ppd, replace=False)
        X_discrete = X_discrete[idx, :]
        X_response = X_response[idx, :]
        outcome = outcome[idx]
    
    log_lik_matrix = np.zeros((samples, N_ppd), dtype=np.float32)
    
    log_lik_insample_list = []
    if parallel != 1:
        res_list = joblib.Parallel(n_jobs=parallel)(joblib.delayed(_k_fold_ELPD_PR_single_thread)(
            X_discrete, 
            X_response, 
            outcome, 
            K,
            D_d,
            D_r,
            i,
            rng_key,
            n_iters,
            samples,
            alpha,
            cluster_gradients
        ) for i in tqdm(split(range(N_ppd), folds), total=folds, position=0, leave=True))
        
        for ind, i in enumerate(split(range(N_ppd), folds)):
            log_lik_matrix[:, i] = res_list[ind][0]
            log_lik_insample_list.append(res_list[ind][1])
        
    else:
        with tqdm(total=folds, position=0, leave=True) as pbar:
            
            for (i, i_inds) in zip(split(range(N_ppd), folds), split(range(N_ppd), folds)):
                log_lik_matrix[:, i_inds], log_lik_insample = _k_fold_ELPD_PR_single_thread(
                    X_discrete, #[idx, :],
                    X_response, #[idx, :],
                    outcome, #[idx],
                    K,
                    D_d,
                    D_r,
                    i,
                    rng_key,
                    n_iters,
                    samples,
                    alpha,
                    cluster_gradients
                )
                log_lik_insample_list.append(log_lik_insample)
                pbar.update(1)
         
    if verbose:
        print("ELPD summary:")
        print("-------------")
        print("Main term: {:.6f}".format((ppd_sampling_factor ** -1) * np.log(np.exp(log_lik_matrix).mean(axis=0)).sum()))
        print("Bias correction: {:.6f}".format(np.log(np.exp(np.asarray(log_lik_insample_list).mean(axis=1)).sum())))
        print("Folds: {}".format(folds))
        print("PPD sampling factor: {}".format(ppd_sampling_factor))
        
    return (ppd_sampling_factor ** -1) * np.log(np.exp(log_lik_matrix).mean(axis=0)).sum() + np.log(np.exp(np.asarray(log_lik_insample_list).mean(axis=1)).sum())




#######################################
# discrete mixture model

@config_enumerate
def discrete_mixture_model(
    K, 
    N, 
    D_discrete, 
    X_discrete=None,
):  

    # priors
    # These are all pretty uninformative priors. Firstly, clusters are equally probable
    cluster_proba = numpyro.sample('cluster_proba', dist.Dirichlet(0.5 * jnp.ones(K))) 
    
    # hyperprior for the phis
    # a beta distribution is symmetric about 0.5 if alpha = beta (common names for the 
    # distribution parameters - there aren't enough Greek letters obviously!
    # Therefore we will draw one number and use that (exponentiated) for alpha and beta
    phi_alpha_beta = numpyro.sample("phi_alpha_beta", dist.Normal(0.0, 1.0))
    
    # Phis (the probabilities in the discrete mixture model)
    # Beta(2,2) is zero at 0 and 1, peaked and symmetric about 0.5
    with numpyro.plate('discrete_components', D_discrete):
        with numpyro.plate("discrete_cluster", K):
            phi = numpyro.sample('phi', dist.Beta(jnp.exp(phi_alpha_beta), jnp.exp(phi_alpha_beta))) 
    
    # model sampling
    with numpyro.plate('data', N):
        
        # Assignment is which cluster each row of data belongs to.
        assignment = numpyro.sample('assignment', dist.CategoricalProbs(cluster_proba)) 
        
        # 1) Discrete mixture model
        # At the moment this assumes independent probabilities
        obs1 = numpyro.sample(
            'obs1', 
            dist.Bernoulli(phi[assignment, :]).to_event(1),
            obs=X_discrete if X_discrete is not None else None,
        )


@partial(jax.jit, static_argnames=["K", "n_iters", "samples"])
def _fit_discrete_mixture_model_SVI_jit(
    X_discrete, 
    K, 
    rng_key,
    n_iters,
    samples
):
    
    N, D = X_discrete.shape
    
    guide = numpyro.infer.autoguide.AutoNormal(
        numpyro.handlers.block(
            numpyro.handlers.seed(discrete_mixture_model, rng_key),
            hide_fn=lambda site: site["name"] in ["assignment"]
        )
    )
    svi = numpyro.infer.svi.SVI(
        discrete_mixture_model,
        guide,
        numpyro.optim.Adam(step_size=0.005),
        numpyro.infer.TraceEnum_ELBO(),
        K=K,
        N=N,
        D_discrete=D,
        X_discrete=X_discrete
    )

    init_state = svi.init(rng_key)
    state, loss = jax.lax.scan(lambda x, i: svi.update(x), init_state, jnp.zeros(n_iters))  
    params = svi.get_params(state)
    
    posterior_predictive_function = numpyro.infer.Predictive(
        discrete_mixture_model, 
        guide=guide, 
        params=params, 
        num_samples=samples
    )
    rng_key, rng_subkey = jax.random.split(key=rng_key)
    posterior_predictive_samples = posterior_predictive_function(
        rng_subkey, 
        K=K,
        N=samples, 
        D_discrete=D,
    )
    
    posterior_function = numpyro.infer.Predictive(
        guide,
        posterior_predictive_samples,
        params=params,
        num_samples=samples
    )
    posterior_samples = posterior_function(
        rng_key,
        K=K,
        N=samples,
        D_discrete=D,
    )
    
    return posterior_samples, posterior_predictive_samples, loss



def fit_discrete_mixture_model(
    X_discrete, 
    K, 
    n_iters,
    samples
):

    """
    Fits a discrete mixture model using SVI and returns an arviz data 
    object and an array recording the loss at each step. 
    
    Parameters
    -----------------------------
        X_discrete: numpy.array
            The discrete data for use in fitting the model. This data
            should be N * D dimensional, where N is the number of individuals
            and D is the number of variables. 
            
        K: int
            The number of clusters.
            
        n_iters: int
            The number of iterations to use in the SVI fitting
            
        samples: int
            The number of posterior samples to compute
        
    """
    
    rng_key = jax.random.PRNGKey(0)
    
    N, D = X_discrete.shape
    
    posterior_samples, posterior_predictive_samples, loss = _fit_discrete_mixture_model_SVI_jit(
        X_discrete, 
        K, 
        rng_key,
        n_iters,
        samples
    )
    
    posterior_samples["assignment"] = posterior_predictive_samples["assignment"]
    
    idx = np.random.choice(N, samples, replace=False)
    log_lik = numpyro.infer.util.log_likelihood(
        discrete_mixture_model, 
        posterior_samples,
        K=K,
        N=samples, 
        D_discrete=D,
        X_discrete=X_discrete[idx, :]
    )
    
    posterior_predictive_dict = {k: v.reshape( [1] + list(v.shape)) for k, v in posterior_predictive_samples.items()}
    inf_dict = {k: v.reshape( [1] + list(v.shape)) for k, v in posterior_samples.items() if k != "assignment"}
    out = az.from_dict(
        inf_dict, 
        posterior_predictive=posterior_predictive_dict,
        observed_data={"obs1":X_discrete},
        log_likelihood={"obs":(log_lik["obs1"]).reshape([1] + list(log_lik["obs1"].shape))}
    )
    
    return out, loss

# ELPD tools

def _k_fold_ELPD_DMM_single_thread(
    X_discrete,
    K,
    D,
    i,
    rng_key,
    n_iters,
    samples
):

    posterior_samples, posterior_predictive_samples, loss = _fit_discrete_mixture_model_SVI_jit(
        np.delete(X_discrete, i, axis=0), 
        K, 
        rng_key,
        n_iters,
        samples
    )

    idx = np.random.choice(samples, len(i), replace=False)
    posterior_samples["assignment"] = np.asarray(posterior_predictive_samples["assignment"])[:, idx]

    log_lik = numpyro.infer.util.log_likelihood(
        discrete_mixture_model, 
        posterior_samples,
        N=len(i), 
        K=K,
        D_discrete=D,
        X_discrete=X_discrete[i, :]
    )

    log_lik_key = list(log_lik.keys())
    
    return log_lik[log_lik_key[0]]
    

    
def k_fold_ELPD_discrete_mixture(
    X_discrete,
    K, 
    n_iters,
    samples,
    folds=None,
    parallel=True
):
    
    """
    Estimates the ELPD of a discrete mixture model using cross validation. The 
    model is fit using SVI. Returns the ELPD estimate.
    
    Parameters
    -----------------------------
        X_discrete: numpy.array
            The discrete data for use in fitting the model. This data
            should be N * D dimensional, where N is the number of individuals
            and D is the number of variables. 
            
        K: int
            The number of clusters.
            
        n_iters: int
            The number of iterations to use in the SVI fitting
            
        samples: int
            The number of posterior samples to compute
            
        folds: int, (default N)
            The number of folds to divide the data into for cross validation. This 
            is equal to the number of times the model must be fitted.
            
        parallel: bool, (default True)
            Whether to split the fitting of the folds across multiple processes.
            Performance may not improve using parallel processing if the data and the
            number of folds are both relatively small. 
        
    """
    
    N, D = X_discrete.shape
    
    if folds is None:
        folds = N
    
    rng_key = jax.random.PRNGKey(0)
    log_lik_matrix = np.zeros((samples, N), dtype=np.float32)

    
    if parallel:
        log_lik_list = joblib.Parallel(n_jobs=-1)(joblib.delayed(_k_fold_ELPD_DMM_single_thread)(
            X_discrete,
            K,
            D,
            i,
            rng_key,
            n_iters,
            samples
        ) for i in tqdm(split(range(N), folds), total=folds, position=0, leave=True))

        for ind, i in enumerate(split(range(N), folds)):
            log_lik_matrix[:, i] = log_lik_list[ind]
    else:
        with tqdm(total=folds, position=0, leave=True) as pbar:
            for i in split(range(N), folds):
                log_lik_matrix[:, i] = _k_fold_ELPD_DMM_singe_thread(
                    X_discrete,
                    K,
                    D,
                    i,
                    rng_key,
                    n_iters,
                    samples,
                )
                pbar.update(1)
                
    return np.log(np.exp(log_lik_matrix).mean(axis=0)).sum()



#######################################
# Multilevel logistic regression model

def multilevel_logistic_model( 
    N, 
    D_response,
    assignment,
    X_response=None,
    y=None
    
):  
    if y is not None:
        y = jnp.array(y)
    
    assignment = assignment.astype(int)
    
    K = len(np.unique(assignment))
    
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0.0, 1.0))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.Normal(1.0))
    mu_beta = numpyro.sample("mu_beta", dist.Normal(0.0, 1.0))
    sigma_beta = numpyro.sample("sigma_beta", dist.Normal(1.0))
    
    with numpyro.plate("response_components", D_response):
        beta = numpyro.sample('beta', dist.Normal(mu_beta, jnp.exp(sigma_beta / 2)))
    
    with numpyro.plate("response_cluster", K):
        intercepts = numpyro.sample('intercepts', dist.Normal(mu_alpha, jnp.exp(sigma_alpha / 2)))

    # model sampling
    with numpyro.plate('data', N):
        
        # There is a separate intercept for each cluster, but only one global gradient.
        # We could have a gradient per cluster if we want.
        linear_predictor = intercepts[assignment] + jnp.sum(beta * X_response, axis=-1)
        obs1 = numpyro.sample(
            "obs1",
            dist.Bernoulli(logits=linear_predictor),
            obs=y if y is not None else None
        ) 
     

if __name__ == "__main__":
    
    print("There should be no reason to run this file directly.")
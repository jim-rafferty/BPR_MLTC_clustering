
import narwhals as nw
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.funsor import config_enumerate
import xarray as xr

import jax
import jax.numpy as jnp
import numpy as np

import arviz as az

import contextlib

# local imports
from .fitting import NUTS, SVI, SteinVI
from .utils import sigmoid, mix_weights


class ProfileRegressionModel(object):

    """
    This class is for conducting Profile Regression Modelling
    
    Fit a model as follows:
    >>> model = ProfileRegressionModel()
    >>> model.fit(X_mixture, X_response, y, K)

    Attributes
    ----------   
    X_mixture : DataFrame
        The mixture sector data used to fit the model
    
    X_response : DataFrame
        The response sector data used to fit the model
    
    y : Series
        The responses (i.e. the dependent variable) 

    K : int
        The maximum number of clusters allowed during model fitting

    result : arviz.InferenceData
        An arviz container for inference data, once the model has been fitted


    Methods
    -------
    fit(X_mixture, X_response, y, K)
        fit(X_mixture, X_response, y, K) -> None
        Fit a profile regression model.
    
    summary()
        summary() -> DataFrame
        Returns a dataframe containing model fit parameters. This is a 
        simple wrapper for arviz.summary

    
    """
    
    def __init__(self):
        """
        Create a Profile Regression model instance
        Parameters
        ----------
        None

        Returns
        -------
        self
        """
        
        self._X_response = None
        self._X_mixture = None
        self._y = None
        self.K = None
        self.response_distribution = None
        self.link_function = None
        self.result = None
        self.losses = None
        
        return 
        
    @property
    def X_mixture(self):
        if self._X_mixture is None:
            return None
        return nw.to_native(self._X_mixture)

    @property
    def X_response(self):
        if self._X_response is None:
            return None
        return nw.to_native(self._X_response)

    @property
    def y(self):
        if self._y is None:
            return None
        return nw.to_native(self._y)

    
    def _model(
        self,
        K, 
        N,
        D_discrete, 
        D_response,
        X_response=None,
        X_mixture=None,
        y=None,
        alpha=None,
        cluster_gradients=False,
        response_distribution=dist.Bernoulli,
        link_function=sigmoid,
        N_batch=None,
    ):  
        """
        Numpyro model function with marginalized discrete latent variable.
        """

        if y is not None:
            y = jnp.array(y)
        
        if X_mixture is not None:
            X_mixture = jnp.array(X_mixture)

        if X_response is not None:
            X_response = jnp.array(X_response)

        # priors
        if alpha is None:
            alpha = numpyro.sample("alpha", dist.Uniform(0.3, 10.0))
        
        with numpyro.plate("v_plate", K-1):
            v = numpyro.sample("v", dist.Beta(1, alpha))
            
        cluster_probabilities = mix_weights(v)
        cluster_probabilities = numpyro.deterministic(
            "cluster_proba", 
            cluster_probabilities, 
        )
            
        # Phis (the probabilities in the discrete mixture model)
        with numpyro.plate('discrete_components', D_discrete):
            with numpyro.plate("cluster_K", K):
                phi = numpyro.sample(
                    "phi", 
                    dist.Uniform(0.001, 0.999) 
                )
        
        # priors for the response model
        with contextlib.ExitStack() as dr_stack:
            if D_response > 0: 
                dr_stack.enter_context(numpyro.plate("response_components", D_response))
                beta = numpyro.sample(
                    'beta', 
                    dist.StudentT(df=7.0, loc=0.0, scale=2.5)
                )
            else:
                beta = 0

        with numpyro.plate("response_cluster", K):
            intercepts = numpyro.sample("intercepts", dist.StudentT(df=7.0, loc=0.0, scale=2.5))

        # model sampling with marginalization
        with numpyro.plate('data', N, subsample_size=N_batch) as ind: 
            
            # Get observed data for this batch
            X_mix_batch = X_mixture[ind] if X_mixture is not None else None
            X_resp_batch = X_response[ind] if X_response is not None else None
            y_batch = y[ind] if y is not None else None
            
            # Compute log likelihoods for each cluster
            log_likelihoods = jnp.zeros((ind.shape[0] if N_batch else N, K))
            
            log_liks = []
            for k in range(K):
                # 1) Discrete mixture log likelihood
                if X_mix_batch is not None:
                    phi_k = phi[k, :]
                    log_lik_discrete = dist.Bernoulli(phi_k).to_event(1).log_prob(X_mix_batch)
                else:
                    log_lik_discrete = 0.0
                
                log_liks.append(log_lik_discrete)
                
                # 2) Response model log likelihood
                if y_batch is not None:
                    linear_pred_k = intercepts[k]
                    if D_response > 0:
                        linear_pred_k = linear_pred_k + jnp.sum(beta * X_resp_batch, axis=-1)
                    
                    log_lik_response = response_distribution(
                        link_function(linear_pred_k)
                    ).log_prob(y_batch)
                else:
                    log_lik_response = 0.0
                
                # Combine likelihoods for cluster k
                log_likelihoods = log_likelihoods.at[:, k].set(
                    log_lik_discrete + log_lik_response
                )
                
            # log cluster probabilities
            log_cluster_probs = jnp.log(cluster_probabilities)
                       
            # Marginalize: log sum_k [pi_k * p(x|z=k)]
            # = log sum_k exp(log pi_k + log p(x|z=k))
            log_prob_marginal = jax.scipy.special.logsumexp(
                log_cluster_probs + log_likelihoods, 
                axis=-1
            )
            
            # Use factor to add the marginalized log likelihood to the model
            numpyro.factor("obs", log_prob_marginal)


    def _calculate_assignments(
        self, 
        phi,
        cluster_probs,
        X
    ): 
        """
        phi: (chains, draws, K, D)
        cluster_probs: (chains, draws, K)
        X: (N, D)
        returns: assignment (chains, draws, N, K)
        """
    
        # Compute log p(x_i | theta_k) for all k
        def log_lik_k(phi_k):
            return (
                dist.Bernoulli(phi_k)
                .to_event(1)
                .log_prob(X)          # (N,)
            )
    
        # Vectorize over K
        def per_draw(phi_draw, cluster_probs_draw):
            # phi_draw: (K, D)
            # cluster_probs_draw: (K,)
    
            assign_log_lik = jax.vmap(log_lik_k)(phi_draw)  # (K, N)
            assign_log_lik = assign_log_lik.T               # (N, K)
    
            log_cluster_probs = jnp.log(cluster_probs_draw + 1e-12)  # (K,)
            assign_log_lik_norm = assign_log_lik + log_cluster_probs[None, :]  # (N, K)
    
            log_assignment = (
                assign_log_lik_norm
                - jax.scipy.special.logsumexp(assign_log_lik_norm, axis=-1, keepdims=True)
            )
    
            assignment = jnp.exp(log_assignment)  # (N, K)
            return assignment
    
        # Vectorize over draws, then chains
        assignment = jax.vmap(          # over chains
            jax.vmap(per_draw, in_axes=(0, 0)),  # over draws
            in_axes=(0, 0),
        )(phi, cluster_probs)
    
        return assignment


        
    @nw.narwhalify
    def fit(
        self,
        X_mixture,
        X_response,
        y,
        K=10,
        method="NUTS",
        samples=1000,
        warmup=1000,
        chains=4,
        steps=10000,
        learning_rate=0.01,
        num_particles=5,
        batch_size = None,
        verbose=True,
        log_lik=False,
        low_rank_mvn = None,
    ):
        """
        Fit a profile regression model.
        
        Parameters
        ----------
        X_mixture : DataFrame
            The observations / independent variables for the mixture sector of the model.
            These observations are used in the Dirichlet process mixutre sector of the 
            profile regression model and determine the clusters present in the data.

        X_response : DataFrame
            The observations / independent variables for the response sector of the model.
            These observations are used in the Generalised Linear Model sector of the model 
            to regress against the outcome / dependent variable.

        y : Series
            The outcome / dependent variable for the reponse sector of the model.
            These observations represent the outcome of interest in the Generalised Linear
            Regression sector of the model.

        K : Int
            The maximum number of clusters permitted in the mixture sector of the model.
            Since we cannot model infinitely many clusters, as would be allowed in a true 
            Dirichlet process mixture model, K specifies the maximum number of allowed 
            clusters (default = 10)

        cluster_gradients: list
            Currently not supported so changing this argument does nothing
            
        method: string
            The method to use to fit the model. Allowed values are "NUTS" or "SVI".

        samples: int
            The number of samples to draw from the posterior distribution (applies to both fit
            methods, following burn-in for NUTS fitting). default = 1000

        warmup: int
            The number of burn-in samples to discard from initial posterior draws. The total 
            number of samples drawn is warmup + samples (NUTS only)

        chains: int
            The number of Markov chains to draw from (NUTS only). Chains can be fitted in parallel
            by running: 
                ```
                import numpyro
                numpyro.set_host_device_count(chains)
                ```
            or using the GPU version of numpyro.

        steps: int
            The number of optimisation steps to perform (SVI only). Default = 10000

        learning_rate: float
            The step size to use in the Adam optimiser (SVI only). Default = 0.001

        low_rank_mvn: int
            The rank of the covariance matrix in the multivariate normal variational
            distribution. Set to 1 for mean-field SVI (SVI only) Default = None (full rank)

        verbose: bool
            Whether to print output to the command window or not. Default = True

        log_lik: bool
            Whether to compute the log likelihood (when using SVI fit). This can be quite 
            memory intensive as the posterior predictive needs to be used. 

        Returns
        -------
        None

        Internal parameters 
        -------
        alpha - this is the parameter in the Dirichlet process that manages how likely it 
            is for samples to be added to a new cluster. The actual probability depends on 
            alpha and the number of clusters that already exist. This is an internal 
            variable and is probably not useful for model interpretation.
            
        beta - these are the coefficients in the response model. By default there is one 
            coefficient for each variable provided in the response sector data. Be aware 
            that if you want to treat a categorical variable as a factor, you will need to 
            manually create dummy variables. 
            
        phi  - from the mixture sector of the model, these are the probabilities of observing
            the discrete variables in each identified cluster. 
            
        v - these are the coefficients derived from the stickbreaking process. They are internal 
            variables related to the cluster probability and are probably not useful for model 
            interpretation.
        """

        if isinstance(y, list):
            y = nw.new_series(
                name="y", 
                values=y,
                native_namespace=nw.get_native_namespace(X_mixture)
            )
        
        self._X_response = X_response
        self._X_mixture = X_mixture
        self._y = y
        self.cluster_gradients = False #cluster_gradients        
        self.K = K
        
        N, D_discrete = self._X_mixture.shape

        if self._X_response is None:
            D_response = 0
            self._X_response = nw.from_dict(
                data={"No response data supplied":[0]},
                native_namespace=nw.get_native_namespace(X_mixture),
            )
        else:
            _, D_response = self._X_response.shape

        if self._y.dtype in [nw.Float32, nw.Float64]:
            # floating point - Gaussian response
            if verbose:
                print("Found continuous response. Using Gaussian distribution and identity link")
            self.response_distribution = dist.Normal
            self.link_function = lambda x: x
        else:
            # integer response
            if self._y.unique().shape[0] == 2:
                # Logistic regression
                if verbose:
                    print("Found binary response. Using Bernoulli distribution and logit link")
                self.response_distribution = dist.BernoulliLogits
                self.link_function = lambda x: x
            else:
                # Count variable - Poisson for now.
                if verbose:
                    print("Found count response. Using Poisson distribution and exponential link")
                self.response_distribution = dist.Poisson
                self.link_function = jnp.exp


        self.response_names = self._X_response.columns


        if method == "NUTS":
            az_obj = NUTS(
                self._model,
                warmup=warmup,
                samples=samples,
                chains=chains,
                discrete_latent=[],#["assignment"],
                verbose=verbose,
                K=K,
                N=N, 
                D_discrete=D_discrete, 
                D_response=D_response,
                X_response=self._X_response[self.response_names].to_numpy(),
                X_mixture=self._X_mixture.to_numpy(),
                y=self._y.to_numpy(),
                alpha=None,
                response_distribution=self.response_distribution,
                link_function=self.link_function
            )

        elif method == "SVI":
            
            guide = None
            if not(low_rank_mvn is None):
                guide = numpyro.infer.autoguide.AutoLowRankMultivariateNormal(self._model, rank=low_rank_mvn)


            if verbose:
                print(f"Number of parameters: {int(1 + K - 1 + D_discrete * K + D_response + K):d}")


            az_obj = SVI(
                self._model,
                steps=steps,
                samples=samples,
                guide=guide,
                elbo_fn=numpyro.infer.Trace_ELBO,
                discrete_latent=[],#["assignment"], 
                learning_rate=learning_rate,
                particles=num_particles,
                verbose=verbose,
                K=K,
                N=N, 
                N_batch=batch_size,
                D_discrete=D_discrete, 
                D_response=D_response,
                X_response=self._X_response[self.response_names].to_numpy(),
                X_mixture=self._X_mixture.to_numpy(),
                y=self._y.to_numpy(),
                alpha=None,
                response_distribution=self.response_distribution,
                link_function=self.link_function,
                log_lik_compute=log_lik,
            )

        else: 
            raise RuntimeError(f"Fit method {method} not supported. Please use either NUTS or SVI")
            
            
        # Calculate assignments
        soft_assignments = self._calculate_assignments(
            az_obj.posterior.phi.to_numpy(),
            az_obj.posterior.cluster_proba.to_numpy(),
            self._X_mixture.to_numpy()
        )
        
        assignment = np.argmax(soft_assignments, axis=-1)
        
        az_obj.add_groups(
            posterior_individual_level={
                "assignment_proba":soft_assignments,
                "assignment":assignment
            },
            coords={
                "assignment_proba":("chain", "draw", "assignment_proba_dim_0", "assignment_proba_dim_1"),
                "assignment":("chain", "draw", "assignment_dim_0")
            }
        )
        
        # Making sure cluster labelling is correct when the number of chains is > 1.

        if method == "NUTS" and chains > 1:
             
            az_obj = az_obj.copy()
            
            cutoff = 0.75 # TODO - make this more flexible?
            
            chain_res = np.zeros((az_obj.posterior_individual_level.assignment.shape[0], az_obj.posterior_individual_level.assignment.shape[2])) * np.nan
            for chain in range(az_obj.posterior_individual_level.assignment.shape[0]):
                for ind in range(az_obj.posterior_individual_level.assignment.shape[2]):
                    
                    freqs = {}
                    dat = az_obj.posterior_individual_level.assignment[chain, :, ind]
                    
                    for val in np.unique(dat):
                        freqs[val] = np.sum(val == np.asarray(dat)) / len(dat)
                        
                    if (np.asarray(list(freqs.values())) > cutoff).any():
                        chain_res[chain, ind] = list(freqs.keys())[list(freqs.values()).index(np.max(list(freqs.values())))]
                        
            u_res = np.unique(chain_res.T, axis=0)
            mapping = u_res[np.isnan(u_res).sum(axis=1) == 0, :]
    
            adjustment_vars = {"cluster_proba":2, "intercepts":2, "phi": 2}
    
            for k in adjustment_vars:
                for chain in range(az_obj.posterior.chain.shape[0]):
                    inds = np.concatenate((mapping[:, chain], np.arange(az_obj.posterior[k].shape[adjustment_vars[k]])))
                    for i in mapping[:, chain]:
                        inds = np.delete(inds, np.argwhere(inds == i)[-1])
            
                    dim_name = f"{k}_dim_0"
                    az_obj.posterior[k][[chain], :, :] = (az_obj.posterior[k][[chain], :, :]
                        .sel({dim_name: inds})
                        .reset_index(dim_name)
                        .reindex({dim_name: np.arange(len(inds))}))
    
            # Assignment
            for chain in range(1, mapping.shape[1]):
                assignment_mapping = {}
                for label in range(mapping.shape[0]):
                    assignment_mapping[mapping[label, 0].astype(int)] = mapping[label, chain].astype(int)
                
                immut_res = az_obj.posterior_individual_level.assignment[chain, :, :]
                for k in assignment_mapping:
                    az_obj.posterior_individual_level.assignment[chain, :, :] = xr.where(
                        immut_res == assignment_mapping[k],
                        k,
                        az_obj.posterior_individual_level.assignment[chain, :, :]
                    )

        if method == "SVI":
            self.losses = az_obj.sample_stats.losses.to_numpy().reshape(-1)
            del az_obj.sample_stats
            
        self.result = az_obj

    def summary(self, var_names=None):
        """
        Return a summary of a fitted model
        
        Parameters
        ----------
        None

        Returns
        -------
        DataFrame
        """

        if self.result is None:
            return
        
        if var_names is None:
            var_names=["~v", "~assignment", "~_latent_distribution", "~_phi_latent"]

        return az.summary(
            self.result, 
            var_names=var_names,
            hdi_prob=0.95,
        ).rename(
            index={"beta[{}]".format(i):k for i, k in enumerate(self.response_names)}
        ).rename( 
            index={"phi[{}, {}]".format(j, i):"Cluster {}, {}".format(j, k) for j in range(self.K) for i, k in enumerate(self._X_mixture.columns)}
        )
        

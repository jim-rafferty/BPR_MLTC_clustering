#!/usr/bin/env python
# coding: utf-8

import numpyro

import pandas as pd
import numpy as np
import arviz as az
import jax
import sys
import scipy.stats as sst

import bayesian_regression_models as brm

import joblib
from tqdm import tqdm 
import gc
import time

import json

# Simulations run here.

def run_simulation_thread(
    samples,
    n_iters,
    data_params,
    seed,
    method,
    thread_idx
):
    
    true_phi_log_odds = np.log(data_params["phi"] / (1 - data_params["phi"]))
    
    D_d, k = data_params["phi"].shape

    intercept_bias = np.zeros((n_iters, k))
    probability_bias = np.zeros((n_iters, k))
    phi_bias = np.zeros((n_iters, k, D_d))
    phi_estimate = np.zeros((n_iters, k, D_d))
    phi_percentile_estimate = np.zeros((n_iters, k, D_d, 2))
    gradients_bias = np.zeros((n_iters, len(data_params["continuous_var_response_gradients"]) + len(data_params["discrete_var_response_gradients"])))
    gradient_percentile_estimate = np.zeros((n_iters, len(data_params["continuous_var_response_gradients"]) + len(data_params["discrete_var_response_gradients"]), 2))

    loopy = range(seed, seed + n_iters)
    
    for it in loopy:
    
        # Generate data
        data = brm.data.gen_data(
            data_params,
            seed=it
        )

        disc_data = data.loc[:, [k for k in data.keys() if k.startswith("d_var_")]]
        r_data = data.loc[:, [k for k in data.keys() if k.startswith("r_var_")]]
    
        # zero centre response data
        r_data_mean = r_data["r_var_0"].mean(axis=0)
        r_data["r_var_0"] = r_data["r_var_0"] - r_data_mean
    
        # Fit model
        model = brm.ProfileRegressionModel()
        if method == "SVI":

            try_fit = True
            n_attempts = 0
            n_steps = 20000
            n_particles = 10

            
            model.fit(
                disc_data,
                r_data,
                data["outcome"].astype(int),
                K=10,
                method="SVI",
                samples=samples,
                num_particles=n_particles,
                steps=n_steps,
                learning_rate=1e-2,
                log_lik=True,
                #LRVB_step=1.0,
                verbose=False,
                batch_size=data_params["batch_size"]
            )

            
        elif method == "Gibbs":
            model.fit(
                disc_data,
                r_data,
                data["outcome"].astype(int),
                K=10,
                method="NUTS",
                samples=samples,
                warmup=9000,
                chains=1,
                verbose=False
            )
            
        az_obj = model.result
        
        # Find the clusters that have higher probability
        if method == "Gibbs":
            weight_mat = az_obj.posterior.cluster_proba.to_numpy()
        elif method == "SVI":
            weight_mat = az_obj.posterior_predictive.cluster_proba.to_numpy()
        weight_mat = weight_mat.mean(axis=(0, 1))
        
        inds = np.argsort(weight_mat)[::-1]
        sim_data_inds = np.argsort(data_params["cluster_probabilities"])[::-1]
    
        # Calculate Biases
        
        probability_bias[it - seed, :] = (np.log(weight_mat[inds[0:k]] / (1 - weight_mat[inds[0:k]])) 
                                          - (np.log(data_params["cluster_probabilities"][sim_data_inds] / (1 - data_params["cluster_probabilities"][sim_data_inds]))))
        
        est_phi = az_obj.posterior["phi"][:, :, inds[0:k], :].mean(axis=(0, 1)).values

        perc_val = 95
        lower = (100 - perc_val) / 2
        upper = 100 - lower
        
        phi_estimate[it - seed, :, :] = est_phi
        phi_percentile_estimate[it - seed, :, :, 0] = np.percentile(az_obj.posterior["phi"][:, :, inds[0:k], :].values, lower, axis=(0, 1))
        phi_percentile_estimate[it - seed, :, :, 1] = np.percentile(az_obj.posterior["phi"][:, :, inds[0:k], :].values, upper, axis=(0, 1))
        
        est_phi_log_odds = np.log(est_phi / (1 - est_phi))
        phi_bias[it - seed, :, :] =  est_phi_log_odds - true_phi_log_odds[:, sim_data_inds].T
        
        intercept_bias[it - seed, :] = (
            az_obj.posterior["intercepts"][:, :, inds[0:k]].mean(axis=(0, 1)).values
            - data_params["intercepts"][sim_data_inds]
        )
        gradients_bias[it - seed, :] = (az_obj.posterior["beta"].mean(axis=(0, 1)).values
                                        - np.concatenate(
                                            (data_params["continuous_var_response_gradients"], data_params["discrete_var_response_gradients"])
                                        ).T)

        gradient_percentile_estimate[it - seed, :, 0] = np.percentile(az_obj.posterior["beta"].values, lower, axis=(0, 1))
        gradient_percentile_estimate[it - seed, :, 1] = np.percentile(az_obj.posterior["beta"].values, upper, axis=(0, 1))
        del data
        del r_data
        del disc_data 
        del model
        del az_obj
        
        #time.sleep(20)
        
        gc.collect()

    return {
        "probability": probability_bias,
        "phi": phi_bias,
        "phi_estimate": phi_estimate,
        "phi_percentile_estimate": phi_percentile_estimate,
        "intercept": intercept_bias,
        "gradient": gradients_bias,
        "gradient_percentile_estimate":gradient_percentile_estimate
    }


def save_data(full_biases, file_label):

    final_results = {}
    start_dict = True
    for b in full_biases:
        if start_dict:
            for k in b.keys():
                #if not(k == "posterior"):
                final_results[k] = b[k]
            start_dict = False
        else:
            for k in b.keys():
                #if not(k == "posterior"):
                final_results[k] = np.concatenate((final_results[k], b[k]))
                
    for k in final_results.keys():
        with open(f"{file_label}_{k}.npy", "wb") as f:
            np.save(f, final_results[k])
    return final_results


def compute_coverage(
    percentile,
    true_val,
    n_iter
):
    # Round to 3 decimal places
    # Run a smaller number of iterations
    return(
        np.logical_and(
            np.round(true_val, decimals=3) >= np.round(percentile[..., 0], decimals=3), 
            np.round(true_val, decimals=3) <= np.round(percentile[..., 1], decimals=3)
        ).sum(axis=0) / n_iter
    )

def main(iterations, batch_frac, size):

    run_gibbs = False
    run_SVI = True
    
    import time
    print(f"Start time {time.ctime()}")
    print("Python", sys.version)
    
    print("numpyro",numpyro.__version__)
    print("pandas",pd.__version__)
    print("arviz",az.__version__)
    print("numpy",np.__version__)
    print("jax",jax.__version__)


    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = np.random.PCG64(0).state

    samples = 1000
    total_iterations = iterations * size

    print(f"\n{iterations} iterations per thread")
    print(f"{size} threads")
    print(f"{total_iterations} total iterations")

    data_params = brm.data.gen_data_parameters()
    data_params["N"] = 800#0
    data_params["batch_size"] = int(data_params["N"] * batch_frac)
    
    print(f"Batch size: {data_params["batch_size"]}")
    
    data_params["phi"] = np.array([[0.03439811, 0.01229981, 0.05205566, 0.01285194, 0.04826528], # phi roll
       [0.57824497, 0.05596716, 0.02733313, 0.06890164, 0.04342215],
       [0.48224405, 0.84322256, 0.01460624, 0.09892868, 0.0876062 ],
       [0.02951533, 0.84171878, 0.91285458, 0.18219329, 0.08728199],
       [0.11808669, 0.05170003, 0.71270472, 0.59012428, 0.11805326],
       [0.04259227, 0.13302682, 0.08348232, 0.53311286, 0.78065395],
       [0.0478201 , 0.03229495, 0.03751321, 0.05630475, 0.89637232],
       [0.05696895, 0.0809764 , 0.02129755, 0.09975493, 0.06696148],
       [0.02355746, 0.06355267, 0.15545181, 0.02950208, 0.10814816],
       [0.01625847, 0.04418503, 0.07100131, 0.26237845, 0.1096367 ],
       [0.03026723, 0.05650182, 0.02710992, 0.07962158, 0.07520882],
       [0.03578021, 0.06418972, 0.03567088, 0.01042063, 0.05920676]]
    )
    
    
    #np.array(
    #    [[0.03439811, 0.06418972, 0.02710992, 0.26237845, 0.10814816], # No overlapping high probs
    #    [0.57824497, 0.01229981, 0.03567088, 0.07962158, 0.1096367 ],
    #    [0.48224405, 0.05596716, 0.05205566, 0.01042063, 0.07520882],
    #    [0.02951533, 0.84322256, 0.02733313, 0.01285194, 0.05920676],
    #    [0.11808669, 0.84171878, 0.01460624, 0.06890164, 0.04826528],
    #    [0.04259227, 0.05170003, 0.91285458, 0.09892868, 0.04342215],
    #    [0.0478201 , 0.13302682, 0.71270472, 0.18219329, 0.0876062 ],
    #    [0.05696895, 0.03229495, 0.08348232, 0.59012428, 0.08728199],
    #    [0.02355746, 0.0809764 , 0.03751321, 0.53311286, 0.11805326],
    #    [0.01625847, 0.06355267, 0.02129755, 0.05630475, 0.78065395],
    #    [0.03026723, 0.04418503, 0.15545181, 0.09975493, 0.89637232],
    #    [0.03578021, 0.05650182, 0.07100131, 0.02950208, 0.06696148]]
    #)

    full_biases_Gibbs_running = []
    full_biases_SVI_running = []

    for it in tqdm(range(iterations)):
        if run_SVI:
            biases_SVI = joblib.Parallel(n_jobs=size)(joblib.delayed(
                run_simulation_thread
                )(
                    samples,                    
                    1,
                    data_params,
                    seed,
                    "SVI",
                    seed - it * size
                ) for seed in np.arange(size) + (it * size)
            )
            full_biases_SVI_running.append(biases_SVI)

        if run_gibbs:
            biases_Gibbs = joblib.Parallel(n_jobs=size)(joblib.delayed(
                run_simulation_thread
                )(
                    samples,                    
                    1,
                    data_params,
                    seed,
                    "Gibbs",
                    seed - it * size
                ) for seed in np.arange(size) + (it * size) 
            )
            full_biases_Gibbs_running.append(biases_Gibbs)


    sim_data_inds = np.argsort(data_params["cluster_probabilities"])[::-1]
    np.save("true_phi.npy", data_params["phi"][:, sim_data_inds].T)
    
    if run_SVI:
        full_biases_SVI = [x for xs in full_biases_SVI_running for x in xs]
        processed_data = save_data(full_biases_SVI, f"simulation_results_SVI_{data_params["batch_size"]}")
        np.save(
            f"simulation_results_SVI_phi_coverage_{data_params["batch_size"]}.npy", 
            compute_coverage(
                processed_data["phi_percentile_estimate"], 
                data_params["phi"][:, sim_data_inds].T, 
                total_iterations
            )
        )
        np.save(
            f"simulation_results_SVI_gradient_coverage_{data_params["batch_size"]}.npy", 
            compute_coverage(
                processed_data["gradient_percentile_estimate"], 
                np.concatenate(
                    (data_params["continuous_var_response_gradients"], 
                     data_params["discrete_var_response_gradients"])
                ).reshape(-1),
            total_iterations
            )
        )
        
    if run_gibbs:
        full_biases_Gibbs = [x for xs in full_biases_Gibbs_running for x in xs]
        processed_data = save_data(full_biases_Gibbs, "simulation_results_Gibbs")
        np.save(
            "simulation_results_Gibbs_phi_coverage.npy", 
            compute_coverage(
                processed_data["phi_percentile_estimate"], 
                data_params["phi"][:, sim_data_inds].T,
                total_iterations
            )
        )
        np.save(
            "simulation_results_Gibbs_gradient_coverage.npy", 
            compute_coverage(
                processed_data["gradient_percentile_estimate"], 
                np.concatenate(
                    (data_params["continuous_var_response_gradients"], 
                     data_params["discrete_var_response_gradients"])
                ).reshape(-1),
            total_iterations
            )
        )
    
    
    print(f"Stop time {time.ctime()}")

    return

if __name__ == "__main__":

    # To run in terminal:
    # python iterative_simulation_study.py n f
    # n is the number of iterations per thread
    # f is the batch size fraction (optional)

    n_iter = int(sys.argv[1])
    
    n_threads = 4 #16
    
    if len(sys.argv) < 3: 
        batch_frac = 1.0
    else:
        batch_frac = float(sys.argv[2])
        
    main(n_iter, batch_frac, n_threads)


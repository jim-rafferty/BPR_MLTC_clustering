
import numpy as np
import pandas as pd


def logistic(x):
    return 1 / (1 + np.exp(-x))


def gen_data_parameters():

    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = np.random.PCG64(0).state

    # Data params
    N = 8000
    D_d = 12
    K = 5
    samples = 1000

    true_phi = rng.beta(1, 20, size=(D_d, K))
    for idx, k in enumerate([list(range(1, 2 * K + 1))[2*i:2*(i+1)] for i in range(K)]):
        true_phi[k, idx] = rng.beta(3, 1, size=(2,))

    true_intercepts = np.array([-14. , -11.5, -10.5,  -9. ,  -8. ]) 
    true_gradients = np.array([[0.2]]) 
    true_cluster_probs = np.array([0.1, 0.2, 0.3, 0.25, 0.35])
    true_cluster_probs = true_cluster_probs / true_cluster_probs.sum()
    r_means = [45] 
    r_cov = np.array([[10 ** 2]])
    r_bin_max_vals = [1, 5]
    true_gradient_disc = np.array([[-0.5], [-0.3], [-0.6], [-0.9], [-1.2]])

    return(
        {
            "cluster_probabilities": true_cluster_probs,
            "phi": true_phi,
            "intercepts": true_intercepts,
            "continuous_var_response_gradients": true_gradients,
            "discrete_var_response_gradients": true_gradient_disc,
            "continuous_var_means": r_means,
            "continuous_var_covariance": r_cov,
            "discrete_var_max_vals": r_bin_max_vals,
            "N":None
        }
    )


def gen_data(
    parameters: dict,
    seed: int = 0,
    response: str = "binomial",
    cluster_gradient: bool = False,
) -> pd.DataFrame :
    
    phi = parameters["phi"]
    intercepts = parameters["intercepts"]
    gradients = parameters["continuous_var_response_gradients"]
    gradients_discrete = parameters["discrete_var_response_gradients"]
    cluster_probs = parameters["cluster_probabilities"]
    r_mean = parameters["continuous_var_means"]
    r_cov = parameters["continuous_var_covariance"]
    r_bin_max = parameters["discrete_var_max_vals"]
    N = parameters["N"]

    if N is None:
        raise Exception("N must be an integer. Got None")
    

    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = np.random.PCG64(seed).state

    D_d, K = phi.shape
    assert cluster_probs.shape[0] == K
    
    D_r = gradients.shape[0]
    
    if D_r > 1:
    
      assert r_mean.shape[0] == D_r
      assert (r_cov.shape == np.array(D_r)).all()

    D_r_d = sum(r_bin_max) - sum([i > 1 for i in r_bin_max])

    cluster_N = np.round(N * cluster_probs).astype(int)

    X_discrete = np.zeros((int(cluster_N.sum()), D_d))
    X_response = np.zeros((int(cluster_N.sum()), D_r + D_r_d))
    outcome = np.zeros(int(cluster_N.sum()))
    outcome_link_function = np.zeros(int(cluster_N.sum()))
    outcome_linear_predictor = np.zeros(int(cluster_N.sum()))
    cluster_label = np.zeros(int(cluster_N.sum()))

    if cluster_gradient:
        # take the first response variable only
        # TODO - fix this for when D_r > 1
        gradients = np.tile(gradients, K) + np.linspace(-0.1 * gradients[0,0], 0.1 * gradients[0,0], K)
        print(gradients)
    
    running_total = 0
    for idx, n in enumerate(cluster_N):
        X_discrete[running_total:(running_total + n), :] = rng.binomial(
            n=1, 
            p=phi[:, idx], 
            size = (n, D_d)
        )
        X_response[running_total:(running_total + n), 0:D_r] = rng.multivariate_normal(
            r_mean,
            r_cov,
            size=n
        )

        running_discrete = 0
        for d_idx, d in enumerate(r_bin_max):

            data = np.round(
                rng.uniform(-0.5, np.max([1, d - 1]) + 0.5, size=n)
            )

            if d == 1:
                X_response[running_total:(running_total + n), D_r + running_discrete] = data.astype(int)
                running_discrete += 1
            else:
                for j in range(1, d):
                    X_response[running_total:(running_total + n), D_r + running_discrete] = (data == j).astype(int)
                    running_discrete += 1

        if cluster_gradient:
            outcome_linear_predictor[running_total:(running_total + n)] = np.dot(
                X_response[running_total:(running_total + n), :], 
                np.concatenate((gradients[:, idx].reshape(-1, 1), gradients_discrete))
            ).sum(axis = 1) + intercepts[idx]
        else:
            outcome_linear_predictor[running_total:(running_total + n)] = np.dot(
                X_response[running_total:(running_total + n), :], 
                (np.concatenate((gradients, gradients_discrete)) if (D_r_d > 0) else gradients)
            ).sum(axis = 1) + intercepts[idx]
        
        if response == "binomial":
            outcome_link_function[running_total:(running_total + n)] = logistic(
                outcome_linear_predictor[running_total:(running_total + n)]
            )
            outcome[running_total:(running_total + n)] = rng.binomial( 
                n=1,
                p=outcome_link_function[running_total:(running_total + n)],
                size=n
            )

        elif response == "gaussian":
            outcome_link_function[running_total:(running_total + n)] = outcome_linear_predictor[running_total:(running_total + n)]
            outcome[running_total:(running_total + n)] = rng.normal( 
                loc=outcome_link_function[running_total:(running_total + n)],
                scale=1.0,
                size=n
            )
        
        cluster_label[running_total:(running_total + n)] = idx
        
        running_total += n
    

    X_discrete = pd.DataFrame(X_discrete).rename(columns={i: "d_var_{}".format(i) for i in range(D_d)})
    X_response = pd.DataFrame(X_response).rename(columns={i: "r_var_{}".format(i) for i in range(D_r + D_r_d)})


    df = pd.merge(
        X_discrete,
        X_response,
        left_index=True,
        right_index=True
    )

    df["outcome"] = outcome
    df["outcome_link_function"] = outcome_link_function
    df["outcome_linear_predictor"] = outcome_linear_predictor
    df["cluster_label"] = cluster_label
    
    return df

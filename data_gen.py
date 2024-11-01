
import pandas as pd
import numpy as np
import itertools

def logit(x):
    return 1 / (1 + np.exp(-x))

def gen_cluster(N, phi, correlated_data, rng):
    
    out = {}
    
    if correlated_data:
        
        D = np.log2(len(phi)).astype(int)
        
        labels = ["".join(ii) for ii in itertools.product(*np.tile("01", D))]
        c_probs = np.cumsum(phi)
        c_probs = np.insert(c_probs, 0, 0.0)

        item_assignment = rng.rand(N) #np.random.rand(N)
        data = np.zeros((N, D))

        for l, d in zip(labels, range(2**D)):
            data[
               ((item_assignment > c_probs[d]) & (item_assignment < c_probs[d+1])),
                :
            ] = np.array([*l], dtype=int)
        
        for d in range(D):
            out["disc_var_{}".format(d)] = data[:, d]
        
    else:
        for d in range(len(phi)):
            out["disc_var_{}".format(d)] = rng.binomial(size=N, n=1, p=phi[d]) # np.random.binomial(size=N, n=1, p=phi[d])
    
    return pd.DataFrame(out)

def gen_data(
    N: int,
    K: int=4, # num clusters
    D: int=8, # num discrete vars
    R: int=4,  # num reponse vars
    n_features_factor: float=1.0, # for tuning the number of features per cluster.
    prob_cutoff: float=0.0,# the lower limit for cluster probabilities (prop to number of people in cluster) 
    empty_cluster: bool=False, # whether or not to include a cluster containing people with no high prob features 
    empty_cluster_with_outcome: bool=False,# whether or not to include a cluster containing people with no high prob features with outcomes
    cluster_gradients: bool=False, # wether or not to define different gradients by cluster
    correlated_data: bool=False,
    seed: int=0
):
    
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = np.random.PCG64(seed).state

    
    if empty_cluster:
        K = K - 1
    
    if empty_cluster_with_outcome:
        K = K - 1
    
    loop_count = 0
    n_clusters_feature = np.array([-1])
    while True:
        
        n_clusters_feature = rng.poisson(n_features_factor * D / K , size=K) #np.random.poisson(n_features_factor * D / K , size=K)
        
        
        if n_clusters_feature.sum() <= D and (n_clusters_feature > 0).all():
            break
        
        if loop_count > 10:
            raise Exception("n_clusters_feature loop: There shouldn't be this many iterations required here.")
            
        loop_count += 1
    
    # discrete data
    inds_full = np.arange(D)
    rng.shuffle(inds_full) # np.random.shuffle(inds_full)
    inds = []
    running_tot = 0
    for i in n_clusters_feature:
        inds.append(inds_full[running_tot:(running_tot+i)])
        running_tot += i
    
    # cluster probs 
    scale_factor = 4 # the larger this number is the less probable very small or very large clusters are
    # and the more uniform the clusters are
    
    
    if empty_cluster and empty_cluster_with_outcome:
        K_prob = K + 2
    elif empty_cluster or empty_cluster_with_outcome:
        K_prob = K + 1
    else:
        K_prob = K
    
    cluster_probs = np.array([-1.0])
    loop_count = 0
    while True:
        
        if K_prob == 1:
            
            cluster_probs = np.array([1.0])
        else:
            cluster_probs = rng.beta(scale_factor, scale_factor*(K_prob-1), size=K_prob-1) # np.random.beta(scale_factor, scale_factor*(K_prob-1), size=K_prob-1)
            cluster_probs = np.append(cluster_probs, 1 - cluster_probs.sum())
        
        if ((cluster_probs * N).round().sum() == N) and ((cluster_probs > prob_cutoff).all()):
            break
        
        if loop_count > 10:
            raise Exception("""closter_probs_loop: There shouldn't be this many iterations required here.\n
            If you have set a probability cutoff, check to make sure the generation is possible with this constraint""")
            
        loop_count += 1
    
    if correlated_data:
        phi_init = rng.dirichlet(rng.rand(2**D), size=K_prob).T # np.random.dirichlet(np.random.rand(2**D), size=K_prob).T

        diff_val = 0.0
        phi = None
        
        for j in range(1, phi_init.shape[1]):
            for i in range(phi_init.shape[0]):
                dv_loop = (np.diff(phi_init, axis=1) ** 2).sum()
                if dv_loop > diff_val:
                    phi = phi_init.copy()
                    diff_val = dv_loop
                phi_init[:, j] = np.roll(phi_init[:, j], 1)

        if phi is None:
            print("phi difference maximsation failed. If K > 1 there is a problem.")
            phi = phi_init
                
        labels = ["".join(ii) for ii in itertools.product(*np.tile("01", D))]
    else:
        labels = range(D)
        phi = rng.beta(1, 10, size=(D, K_prob)) #np.random.beta(1, 10, size=(D, K_prob))
        for k in range(K): # potentially leaving the last cluster with no high probability variables
            for d in inds[k]:
                phi[d, k] = rng.beta(3, 1) #np.random.beta(3, 1)
    
    
    
    boundaries = []
    for k in range(K_prob):
        ind = int(np.round(N * cluster_probs[k]))
        boundaries.append(ind)
        if k == 0:
            df = gen_cluster(ind, phi[:, k], correlated_data, rng)
        else:
            df = pd.concat([df, gen_cluster(ind, phi[:, k], correlated_data, rng)])

    
    phi = pd.DataFrame(phi, index=labels)

    df = df.reset_index()
    df = df.drop(columns="index")
    
    # response data
    
    if cluster_gradients:
        coefficients = rng.normal(0, 1, size=(R, K_prob))
    else:
        coefficients = rng.normal(0, 1, size=(R, 1)) + 1 #np.random.randn(R) + 1
    intercepts = rng.normal(0, 1, size=K_prob) # np.random.randn(K_prob)
    
    if empty_cluster:
        intercepts[-1] = rng.normal(0, 1) - 12 # np.random.randn() - 12
    
    r_data = rng.normal(0, 1, size=(N, R)) #np.random.randn(N, R)
    
    outcome = np.zeros(N)
    cluster_label = np.zeros(N)
    
    running_ind = 0
    coef_ind = 0
    
    for k, b in enumerate(boundaries):
        
        if cluster_gradients:
            coef_ind = k
            
        outcome[running_ind:(running_ind + b)] = rng.binomial( #np.random.binomial(
            size=b,
            n=1,
            p=logit((r_data[running_ind:(running_ind + b), :] * coefficients[:, coef_ind]).sum(axis = 1) + intercepts[k])
        )
        
        cluster_label[running_ind:(running_ind + b)] = k
        
        running_ind += b
    
    df = df.merge(
        pd.DataFrame(r_data).rename(columns={i: "r_var_{}".format(i) for i in range(R)}),
        left_index=True,
        right_index=True
    )
    df["outcome"] = outcome
    df["cluster_label"] = cluster_label # this is to be used for model diagnostics, not fitting
    
    
    # shuffle the rows of the dataframe so that clusters are not in blocks
    df = df.sample(frac=1.0).reset_index(drop=True)
    
    return {
        "dataframe": df,
        "phi": phi,
        "coefficients": coefficients,
        "intercepts": intercepts,
        "cluster_probs": cluster_probs,
        "cluster_features": inds
    }
    
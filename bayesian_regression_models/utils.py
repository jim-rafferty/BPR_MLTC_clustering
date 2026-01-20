import itertools
import pandas as pd
import jax.numpy as jnp

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def mix_weights(beta):
    """
    Function to do the stick breaking construction
    """
    beta1m_cumprod = jnp.cumprod(1 - beta, axis=-1)
    term1 = jnp.pad(beta, (0, 1), mode='constant', constant_values=1.)
    term2 = jnp.pad(beta1m_cumprod, (1, 0), mode='constant', constant_values=1.)
    return jnp.multiply(term1, term2)

def combine_bracketed_terms(lst):
    result = []
    buffer = []
    in_brackets = False

    for item in lst:
        if '(' in item:
            in_brackets = True
            buffer.append(item)
        elif ')' in item:
            buffer.append(item)
            # Join internal terms with ' + ', handling the brackets
            joined = '+'.join(
                s.strip('()') for s in buffer
            )
            result.append(f"({joined})")
            buffer = []
            in_brackets = False
        elif in_brackets:
            buffer.append(item)
        else:
            result.append(item)

    return result



def process_formula(
    f: str, 
    data: pd.DataFrame
):

    resp = f.split("~")[0].strip()
    preds = [i.replace(" ", "") for i in f.split("~")[1].strip().split("+")]
    preds = combine_bracketed_terms(preds)

    # interaction terms
    
    # NOTE 
    # - : denotes an interaction term x:y
    # - * expands to all possible terms, ie x * y = x + y + x:y
    exp_terms = ["*" in i for i in preds] 
    orig_preds = preds.copy()
    
    for i, b in zip(orig_preds, exp_terms):
        if b:
            int_vars = [j.strip() for j in i.split("*")]
            preds.remove(i)
            # NOTE - insufficient checks for higher order terms
            if len(int_vars) > 2:
                for b in range(len(int_vars)-1, 1, -1):
                    for a in itertools.combinations(int_vars, b):
                        if ":".join(a) not in preds and ":".join(a[::-1]) not in preds:
                            preds.append(":".join(a))
            if ":".join(int_vars) not in preds and ":".join(int_vars[::-1]) not in preds:
                preds.append(":".join(int_vars))
            for v in int_vars:
                if v not in preds:
                    preds.append(v)
    

    
    int_terms = [":" in i for i in preds] 
    #for i, b in zip(preds, int_terms):
    #    if b:
    #        int_vars = [j.replace(" ", "") for j in i.split(":")]
            #data[i] = data[int_vars].prod(axis=1)
            
    # mixed effects

    # NOTE | denotes mixed effects.
    # - (1|ID) denotes a random intercept model with a different intercept for each ID
    # - (0 + x|ID) denotes a random gradient model with a different gradient for x for each ID and a global intercept
    # - (1 + x|ID) denotes a random gradient model with a different gradient for x and a different intercept for each ID

    me_terms = ["|" in i for i in preds]

    #for i, b in zip(orig_preds, me_terms):
    #    if b:
    #        print(i)
        
    print(preds)
    print(int_terms)
    print(me_terms)

    # TODO - pull out the indices of the terms in the dataframe
    # Figure out the mixed effects stuff
    # Survival parsing

    return #data.loc[:, preds], data[resp]
import numpy as np
from scipy.optimize import root_scalar
import pandas as pd

def crra_utility(w, gamma):

    if gamma != 1:
        return (w**(1 - gamma) - 1) / (1 - gamma)
    else:
        return np.log(w)

def prelec_weight(p, alpha):

    if p == 0:
        return 0.0
    elif p == 1:
        return 1.0
    else:
        return np.exp(-((-np.log(p))**alpha))

def find_indifference_X(gamma, W0, gA, p_loss, G, alpha=0.5):

    p_gain = 1 - p_loss
    
    W_A = W0 + gA
    U_A = crra_utility(W_A, gamma)
    
    W_gain = W0 + G
    U_gain = crra_utility(W_gain, gamma)
    w_gain = prelec_weight(p_gain, alpha)
    
    # U_A - (w_loss * U_loss + w_gain * U_gain) = 0
    def equation(X):
        W_loss = W0 - X
        if W_loss <= 0:
            # utility undefined for non-positive wealth; return a large positive value to avoid root finding here
            return U_A - (-1e10)
        U_loss = crra_utility(W_loss, gamma)
        w_loss = prelec_weight(p_loss, alpha)
        V_B = w_loss * U_loss + w_gain * U_gain
        return U_A - V_B
    
    # X must be between 0 and W0 to ensure W_loss > 0
    lower_X = 0.0
    upper_X = W0 * 0.99  # slightly less than W0 to avoid W_loss = 0
    
    # root_scalar to find X where equation(X) = 0
    try:
        solution = root_scalar(equation, bracket=[lower_X, upper_X], method='brentq')
        if solution.converged:
            return solution.root
        else:
            return np.nan
    except ValueError:
        return np.nan

def compute_indifference_across_alphas(gamma, W0, gA, p_loss, G, alpha_values):

    results = []
    for alpha in alpha_values:
        X = find_indifference_X(gamma, W0, gA, p_loss, G, alpha)
        results.append({'Alpha': alpha, 'Indifference Loss Amount (X)': X})
    df = pd.DataFrame(results)
    return df

# Example usage
if __name__ == "__main__":
    # Define parameters
    gamma = 3       # Rrelative risk aversion coefficient (gamma=1 for log utility)
    W0 = 100.0         # starting wealth
    gA = 1.0          # guaranteed gain in choice A (resulting in $11)
    p_loss = 0.98     # probability of loss in choice B
    G = 98.0          # gain prospect in choice B
    
    # alpha values from 0 to 1 in increments of 0.1
    alpha_values = np.arange(0.0, 1.05, 0.05)
    
    indifference_df = compute_indifference_across_alphas(gamma, W0, gA, p_loss, G, alpha_values)
    
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print(indifference_df.to_string(index=False))

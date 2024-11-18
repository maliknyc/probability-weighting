import numpy as np
from scipy.optimize import root_scalar
import pandas as pd
import matplotlib.pyplot as plt

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
        w_loss = 1 - w_gain
        V_B = w_loss * U_loss + w_gain * U_gain
        return U_A - V_B
    
    # X must be between 0 and W0 to ensure W_loss > 0
    lower_X = 0.0
    upper_X = W0 * 0.99  # Slightly less than W0 to avoid W_loss = 0
    
    # root_scalar to find X where equation(X) = 0
    try:
        solution = root_scalar(equation, bracket=[lower_X, upper_X], method='brentq')
        if solution.converged:
            return solution.root
        else:
            return np.nan  # return NaN if doesn't converge
    except ValueError:
        return np.nan  # return NaN if no root found in interval

def compute_indifference_across_alphas(gamma, W0, gA, p_loss, G, alpha_values):

    p_gain = 1 - p_loss

    W_A = W0 + gA
    U_A = crra_utility(W_A, gamma)
    
    results = []
    for alpha in alpha_values:
        X = find_indifference_X(gamma, W0, gA, p_loss, G, alpha)
        if np.isnan(X):
            V_B = np.nan
            E_choice_B = np.nan
        else:
            V_B = U_A  # by indifference condition
            E_choice_B = p_loss * (W0 - X) + p_gain * (W0 + G)
        results.append({
            'Alpha': alpha,
            'Indifference Loss Amount (X*)': X,
            'Expected Utility of Risky Choice (V_B)': V_B,
            'Expected Value of Risky Choice': E_choice_B
        })
    df = pd.DataFrame(results)
    return df

def compute_and_plot_indifference(gamma, W0, gA, p_loss, G, alpha_values, example_number, color):

    indifference_df = compute_indifference_across_alphas(gamma, W0, gA, p_loss, G, alpha_values)
    
    # EU of non-risky choice
    W_A = W0 + gA
    U_A = crra_utility(W_A, gamma)
    print(f"\nConfiguration {example_number}: Expected Utility of Non-Risky Choice (U_A) = {U_A:.4f}")
    
    indifference_df['Indifference Loss Amount (X*)'] = indifference_df['Indifference Loss Amount (X*)'].round(2)
    indifference_df['Expected Value of Risky Choice'] = indifference_df['Expected Value of Risky Choice'].round(2)
    indifference_df['Expected Utility of Risky Choice (V_B)'] = indifference_df['Expected Utility of Risky Choice (V_B)'].round(4)
    
    print(f"\nConfiguration {example_number}: Indifference Loss Amounts Across Alphas")
    print(indifference_df.to_string(index=False))
    
    # plot for X* vs. Alpha
    plt.plot(
        indifference_df['Alpha'],
        indifference_df['Indifference Loss Amount (X*)'],
        marker='o',
        linestyle='-',
        label=f'Config {example_number}',
        color=color
    )


if __name__ == "__main__":
    # define alphas from 0.0 to 1.0 in increments of 0.1
    alpha_values = np.arange(0.05, 1.05, 0.05)
    
    '''
    gamma1 = 0       # relative risk aversion coefficient (gamma=1 for log utility)
    W0_1 = 1000         # starting wealth
    gA1 = 0          # guaranteed gain in choice A
    p_loss1 = 0.98     # probability of loss in choice B
    G1 = 1000          # gain prospect in choice B
    configuration = 1
    color1 = 'black'    
    
    compute_and_plot_indifference(gamma1, W0_1, gA1, p_loss1, G1, alpha_values, example_number=configuration, color=color1)
    

    gamma1 = 1       # relative risk aversion coefficient (gamma=1 for log utility)
    W0_1 = 1000         # starting wealth
    gA1 = 0          # guaranteed gain in choice A 
    p_loss1 = 0.98     # probability of loss in choice B
    G1 = 500000          # gain prospect in choice B
    configuration = 2
    color1 = 'blue'    
    
    compute_and_plot_indifference(gamma1, W0_1, gA1, p_loss1, G1, alpha_values, example_number=configuration, color=color1)

  

    gamma1 = 2       # relative risk aversion coefficient (gamma=1 for log utility)
    W0_1 = 1000        # rtarting wealth
    gA1 = 0          # guaranteed gain in choice A
    p_loss1 = 0.98     # probability of loss in choice B
    G1 = 5000          # gain prospect in choice B
    configuration = 3
    color1 = 'green'   
    
    compute_and_plot_indifference(gamma1, W0_1, gA1, p_loss1, G1, alpha_values, example_number=configuration, color=color1)
    

    gamma1 = 3       # relative risk aversion coefficient (gamma=1 for log utility)
    W0_1 = 1000         # starting wealth
    gA1 = 0          # guaranteed gain in choice A
    p_loss1 = 0.98     # probability of loss in choice B
    G1 = 1000          # gain prospect in choice B
    configuration = 4
    color1 = 'brown'    
    
    compute_and_plot_indifference(gamma1, W0_1, gA1, p_loss1, G1, alpha_values, example_number=configuration, color=color1)
    '''
    gamma1 = 0.5       # relative risk aversion coefficient (gamma=1 for log utility)
    W0_1 = 1000         # starting wealth
    gA1 = 0          # guaranteed gain in choice A
    p_loss1 = 0.98     # probability of loss in choice B
    G1 = 1000          # gain prospect in choice B
    configuration = gamma1
    color1 = 'blue'    
    
    compute_and_plot_indifference(gamma1, W0_1, gA1, p_loss1, G1, alpha_values, example_number=configuration, color=color1)
    
    gamma1 = 1       # relative risk aversion coefficient (gamma=1 for log utility)
    W0_1 = 1000         # starting wealth
    gA1 = 0          # guaranteed gain in choice A
    p_loss1 = 0.98     # probability of loss in choice B
    G1 = 1000          # gain prospect in choice B
    configuration = gamma1
    color1 = 'black'    
    
    compute_and_plot_indifference(gamma1, W0_1, gA1, p_loss1, G1, alpha_values, example_number=configuration, color=color1)
    
    gamma1 = 1.5       # relative risk aversion coefficient (gamma=1 for log utility)
    W0_1 = 1000         # starting wealth
    gA1 = 0          # guaranteed gain in choice A
    p_loss1 = 0.98     # probability of loss in choice B
    G1 = 1000          # gain prospect in choice B
    configuration = gamma1
    color1 = 'purple'    
    
    compute_and_plot_indifference(gamma1, W0_1, gA1, p_loss1, G1, alpha_values, example_number=configuration, color=color1)

    gamma1 = 2       # relative risk aversion coefficient (gamma=1 for log utility)
    W0_1 = 1000         # starting wealth
    gA1 = 0          # guaranteed gain in choice A
    p_loss1 = 0.98     # probability of loss in choice B
    G1 = 1000          # gain prospect in choice B
    configuration = gamma1
    color1 = 'red'    
    
    compute_and_plot_indifference(gamma1, W0_1, gA1, p_loss1, G1, alpha_values, example_number=configuration, color=color1)

    # finalize and display the plot
    plt.title('Indifference Loss Amount (X*) vs. Prelec\'s Alpha (α)')
    plt.xlabel('Prelec\'s Alpha (α)')
    plt.ylabel('Indifference Loss Amount (X*)')
    plt.grid(True)
    plt.legend()
    plt.xticks(alpha_values*2)
    plt.xlim(0,1)
    plt.show()



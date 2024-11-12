## CRRA Utility Function

$$
U(w) =
\begin{cases} 
\frac{w^{1 - γ} - 1}{1 - γ} & \text{for } γ \neq 1 \\
\ln(w) & \text{for } γ = 1
\end{cases}
$$

**Where:**
- \( w \) is wealth.
- \( γ \) is the relative risk aversion coefficient.

## Prelec's Probability Weighting Function

$$
w(p) = \exp\left(-β \cdot \left(-\ln(p)\right)^\alpha\right), \quad \text{for } 0 < p < 1
$$

**Where:**
- \( w(p) \) is the **perceived probability** corresponding to the **objective probability** \( p \).
- \( p \) is the **objective probability** of an event occurring.
- \( α \) is the **curvature parameter** (\( 0 < \alpha \leq 1 \)):
  - \( α = 1 \): The function becomes the identity function, \( w(p) = p \), indicating **linear probability perception**.
  - \( α < 1 \): The function exhibits **inverse-S shaped** behavior, **overweighting** small probabilities and **underweighting** large probabilities.
- \( β \) is the **scaling parameter** introduced to shift the indifference point.
 
## Modified Probability Weighting Function

$$
w(p) = \exp\left(-(\ln 2)^{1 - \alpha} \cdot \left(-\ln(p)\right)^\alpha\right), \quad \text{for } 0 < p < 1
$$
  
**Derivation:**

Set point where w(p) = p at p = 0.5 and solve for β:

  $$
  \exp\left(-β \cdot \left(-\ln(0.5)\right)^\alpha\right) = 0.5
  $$

  $$
  \exp\left(-β \cdot (\ln 2)^\alpha\right) = 0.5
  $$

  $$
  -β \cdot (\ln 2)^\alpha = \ln(0.5) = -\ln 2
  $$

  $$
  β = \frac{\ln 2}{(\ln 2)^\alpha} = (\ln 2)^{1 - \alpha}
  $$
  
Plug β back into Prelec's function to obtain final form.

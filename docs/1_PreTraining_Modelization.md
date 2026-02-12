# I. Portfolio modelization
*Corresponding modules :* `portfolio_model.py` 

## 1. Portfolio dynamics
To begin, we are given a finite timeline from $t=0$ to $t=T$ discretized into $N$ equal periods. We assume that there is a fixed selection of $d$ assets $S = (S^1, ...S^d)$ to be included in the portfolio whose dynamics (type of model + model parameters) are known. This, at any given moment $t$, the current state of the portfolio is represented by a state variable $X_t = (S_t, r_t, \beta_t, \phi_t, L_t)$ where $S$ is the market value of the assets,  $r$ the risk-free interest rate, $\beta$ the cash amount in the portfolio, $\phi = (\phi^1,...\phi^d)$ the quantity held of each asset, and $L$ the *Mathematical Provision* of the life insurance portfolio (which is a liability). We suppose that the coupon rates $c$ and the book values $\tilde S$ of the assets are deterministic and known since the beginning, and hence not included as a part of the state variable $X$.  

The manager's control $u_t$ to be applied at $t$ comprises of the trade amount $\dot \phi_t$ (reallocation) and the proportion of profit sharing $\pi_t$ (distribution). To be admissible, the control must be within a given range. From here, we define the dynamic of the portfolio state in a discrete framework as follow
$$ 
X_{t_{i+1}} = X_{t_i} + b(X_{t_i}, u_{t_i}) \Delta t + \sigma(X_{t_i}) \Delta W_{t_{i+1}} 
$$ 
where $\Delta W_{t_{i+1}} := W_{t_{i+1}} - W_{t_i}$ is given from a multivariate brownian movements $W$ which comprises of $d+1$ independent brownians that drive the dynamics of the $d$ assets and the interest rate $r$. More precisely, we define 
$$
b(X_t,u_t) = \begin{bmatrix}
        \mu_S S_t \\ 
        \kappa_r(\mu_r - r_t)\\
        \beta_t r_t + \phi_t \dot{c_t} - \dot \gamma_t L_t - \dot{\phi_t} \cdot S_t\\
        \dot \phi_t \\
        \pi_t \dot \eta_t - \dot \gamma_t L_t 
    \end{bmatrix}
    \text{ and }
    \sigma(X_t, u_t) = \begin{bmatrix}
        \sigma_S S_t & 0 \\
        0 & \sigma_r\\
        0 & 0\\
        0 & 0\\
        0 & 0
    \end{bmatrix}
$$
with $\dot \eta_t := \beta_t r_t + \phi_t \dot c_t + (\dot \phi_t)^- \cdot (S_t - \tilde S_t)$ is the financial production intensity (following the reallocation of assets) and $\dot \gamma_t := \gamma^0 + \gamma^1(r_t - \phi_t \dot \eta_t)$ is the lapse intensity (following the sharing of profit) at $t$. 

We assume a uniform distribution for every state variable at $t=0$. 

## 2. Constraints 

We impose some constraints on the portfolio, which are manifested as penalties in the training of the optimal controls $u_\theta$ at the boundary of the viable domain. Notably, the following penalties are applied :
* No-bankruptcy penalty : at every period, the total liability $L$ must be inferior to the total asset $\beta + \phi \cdot S$;  the corresponding penalty function is 
$$
g^1(X_t) := \max(L_t - \beta_t - \phi_t \cdot S_t, 0)
$$
* Non-negative liability penalty : at every period, if the amount of liability ever touches 0, then the portfolio is no longer in business ; the corresponding penalty function is 
$$
g^2(X_t) := \max(- L_t, 0)
$$
* Non-negative cash penalty : at every period, the amount of cash in holding must not be negative (liquidity requirement); the corresponding penalty function is 
$$
g^3(X_t) := \max(- \beta_t, 0)
$$

In additional to the penalties listed above, we also care about the bouned terminal capital loss which is defined mathematically as 
$$
G(X_T, \underline K, \overline K) := \max \{ \underline K, \min \{ \overline K, L_T - \beta_T - \phi_T \cdot S_T \}\}
$$
for some constants $\underline K < \overline K$. Note that as $\underline K \rightarrow -\infty$ and $\overline K \rightarrow \infty$, one retrieves the simple terminal capital loss. We add the bounds $\underline K$ and $\overline K$ for technical reason (referring to the article for more explanation).

## 3. Reward function

Our reward function represents the utility of the portfolio state, and this function is to be computed solely at the end of the time horizon. More specifically in this case, we choose an exponential utility function $F$ over net terminal wealth defined as 
$$ 
F(X_T) = - \alpha * \exp [ - \alpha * (\beta_T + \phi_T \cdot S_T - L_T)] 
$$
where $ \alpha$ is the absolute risk aversion coefficient. 

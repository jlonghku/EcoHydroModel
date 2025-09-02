#  Nitrogen Module: Principles and Governing Equations

## 1 Overview
We implement a lumped, daily‐time‐step nitrogen (N) module coupled to simulated streamflow. The module tracks three conceptual pools—ammonium (NH₄⁺), nitrate (NO₃⁻), and organic nitrogen (OrgN)—and represents (i) external inputs linked to hydrologic forcing, (ii) plant uptake, (iii) mineralization, (iv) temperature-modulated losses, and (v) hydrologic export. A first-order reservoir produces lagged NO₃⁻ signals at the outlet.

## 2 State Variables and Fluxes
Let \(N_{\mathrm{NH4}}(t)\), \(N_{\mathrm{NO3}}(t)\), \(N_{\mathrm{Org}}(t)\) be pool masses (e.g., mg N m\(^{-2}\) or kg N ha\(^{-1}\)). Daily streamflow \(Q(t)\) is provided by the HBV component. Temperature \(T(t)\) drives biological rates.

External N input linked to flow:
\[
I_{\mathrm{NH4}}(t)=\gamma Q(t),\qquad
I_{\mathrm{NO3}}(t)=\gamma Q(t),\qquad
I_{\mathrm{Org}}(t)=1.5\,\gamma Q(t),
\]

Vegetation activity index \(V(t)\) (logistic with thermal optimum) modulates uptake:
\[
V(t)=\frac{1}{1+\exp[-0.5\,(T(t)-15)]}.
\]

A temperature modifier adjusts reaction/exit rates:
\[
f_T(t)=\exp\big[\beta\,(T(t)-10)\big],
\]
where \(\beta\) controls temperature sensitivity.

## 3 Process Representations
**(a) Plant uptake**  
\[
U_{\mathrm{NH4}}(t)=k_{\mathrm{up}}V(t)N_{\mathrm{NH4}}(t),\qquad
U_{\mathrm{NO3}}(t)=k_{\mathrm{up}}V(t)N_{\mathrm{NO3}}(t).
\]

**(b) Mineralization (OrgN → inorganic N)**  
\[
M_{\mathrm{Org}}(t)=k_{\mathrm{dec}}N_{\mathrm{Org}}(t).
\]
Partitioning:  
\[
\Delta N_{\mathrm{NH4}}^{\mathrm{min}}=0.2M_{\mathrm{Org}}(t),\quad
\Delta N_{\mathrm{NO3}}^{\mathrm{min}}=0.3M_{\mathrm{Org}}(t).
\]

**(c) Temperature-modulated hydrologic losses**  
\[
L_{\mathrm{NH4}}(t)=\alpha_{\mathrm{NH4}}Q(t)f_T(t),\quad
L_{\mathrm{NO3}}(t)=\alpha_{\mathrm{NO3}}Q(t)f_T(t),\quad
L_{\mathrm{Org}}(t)=\alpha_{\mathrm{Org}}Q(t)f_T(t).
\]

## 4 Pool Mass Balances (explicit Euler, daily)
\[
\begin{aligned}
N_{\mathrm{NH4}}(t{+}1) &= \max\{0, N_{\mathrm{NH4}}(t) + I_{\mathrm{NH4}}(t) + \Delta N_{\mathrm{NH4}}^{\mathrm{min}} - U_{\mathrm{NH4}}(t) - L_{\mathrm{NH4}}(t)\},\\
N_{\mathrm{NO3}}(t{+}1) &= \max\{0, N_{\mathrm{NO3}}(t) + I_{\mathrm{NO3}}(t) + \Delta N_{\mathrm{NO3}}^{\mathrm{min}} - U_{\mathrm{NO3}}(t) - L_{\mathrm{NO3}}(t)\},\\
N_{\mathrm{Org}}(t{+}1) &= \max\{0, N_{\mathrm{Org}}(t) + I_{\mathrm{Org}}(t) - M_{\mathrm{Org}}(t) - L_{\mathrm{Org}}(t) + 0.1\}.
\end{aligned}
\]

## 5 Outlet Nitrate with Transport Lag
Instantaneous NO₃⁻ export \(E_{\mathrm{NO3}}(t)=L_{\mathrm{NO3}}(t)\) is routed through a first-order linear reservoir:  
\[
Y_{\mathrm{NO3}}(t)=(1-\lambda)Y_{\mathrm{NO3}}(t-1)+\lambda E_{\mathrm{NO3}}(t).
\]

## 6 Parameters and Initial Conditions
\[
\Theta=\{\alpha_{\mathrm{NH4}},\,\alpha_{\mathrm{NO3}},\,\alpha_{\mathrm{Org}},\,k_{\mathrm{up}},\,k_{\mathrm{dec}},\,\beta,\,\gamma,\,N_{\mathrm{NH4}}(0),\,N_{\mathrm{NO3}}(0),\,N_{\mathrm{Org}}(0),\,\lambda\}.
\]

**Typical bounds (used in calibration):**  
- \(\alpha\): [1e−4, 5e−2]  
- \(k_{up}\): [1e−3, 1e−1]  
- \(k_{dec}\): [1e−4, 2e−2]  
- \(\beta\): [0,1]  
- \(\gamma\): [0,0.1]  
- \(N(0)\): [0.1,50]  
- \(\lambda\): [0.01,0.5]  

## 7 Coupling, Numerics, and Calibration
- **Coupling:** \(Q(t)\) and \(T(t)\) are provided by the hydrologic component (HBV). External N inputs scale with \(Q(t)\).  
- **Time stepping:** explicit Euler, daily. Non-negativity is enforced by truncation.  
- **Objectives:** goodness-of-fit to observed stream NO₃⁻ using NSE or KGE, optionally MSE; flow is calibrated first, then N parameters are optimized conditional on simulated flow.  
- **Identifiability:** \(\gamma\), \(\alpha_{\mathrm{NO3}}\), and \(\lambda\) may be correlated through their joint control on timing and magnitude; \(\beta\) interacts with seasonal \(V(t)\).

## 8 Assumptions and Limitations
This is a reduced-form, lumped representation. Nitrification/denitrification are not explicitly separated; mineralization is a single first-order process with fixed partitioning. The first-order routing is a pragmatic proxy for in-stream and near-stream lags. Despite its simplicity, the structure captures primary controls of seasonal uptake, temperature, hydrologic export, and transport delay.

---

### Symbol and Parameter Glossary
- Q(t): streamflow (driver)  
- T(t): temperature (driver)  
- V(t): vegetation activity index  
- f_T(t): temperature modifier  
- α_NH4, α_NO3, α_Org: hydrologic loss coefficients  
- k_up: plant uptake rate  
- k_dec: mineralization rate  
- β: temperature sensitivity  
- γ: external input coefficient  
- N_NH4(0), N_NO3(0), N_Org(0): initial pools  
- λ: routing/lag parameter  
- Y_NO3(t): lagged NO₃⁻ signal at outlet  

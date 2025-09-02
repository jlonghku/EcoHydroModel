# Process-Based Nitrification Module (Del Grosso–type Formulation)

## 1. Overview
We implement a process-based nitrification module following Del Grosso–type response functions. The daily nitrification rate is computed as a **potential term** driven by ammonium availability and humus decomposition, **modulated multiplicatively** by soil water-filled pore space (WFPS), temperature, and pH response functions. Soil-specific coefficients allow heterogeneity across soil classes. This nitrification flux is coupled to the nitrogen pools and hydrologic drivers defined elsewhere in the model.

## 2. Governing Formulation

### 2.1 Potential Term
Let \(N_{\mathrm{NH4}}(t)\) be the NH\(_4^+\) pool and \(H_{\mathrm{dec}}(t)\) the daily amount of N released from humus decomposition. The potential (unlimited) nitrification driver is
$$
B(t) = k_{1}\,H_{\mathrm{dec}}(t) + k_{\max}\,N_{\mathrm{NH4}}(t),
$$
where \(k_{1}\) and \(k_{\max}\) are rate coefficients.

### 2.2 Environmental Limitation Functions

**(a) Water limitation (WFPS):**
$$
F_{w}(w) =
\left(\frac{w-c_{1}}{c_{0}-c_{1}}\right)^{\;c_{3}\frac{(c_{1}-c_{0})}{(c_{0}-c_{2})}}
\;
\left(\frac{w-c_{2}}{c_{0}-c_{2}}\right)^{\;c_{3}},
$$
with \(w\) the water-filled pore space (WFPS) and \(c_{0},c_{1},c_{2},c_{3}\) empirical shape parameters.

**(b) Temperature limitation:**
$$
F_{T}(T) =
\left(\max\!\left\{0,\; \frac{c_{1}-T}{c_{1}-25}\right\}\right)^{c_{2}}
\exp\!\left(
\frac{c_{2}}{c_{3}}\Big[1-\big(\max\!\{0,\tfrac{c_{1}-T}{c_{1}-25}\}\big)^{c_{3}}\Big]
\right),
$$
where \(T\) is soil temperature (°C) and \(c_{1},c_{2},c_{3}\) control the curve shape and optimum.

**(c) pH limitation:**
$$
F_{\mathrm{pH}}(\mathrm{pH}) =
\min\!\left\{1,\; c_{1}+\frac{c_{2}}{\pi}\arctan\!\big(\pi c_{3}(\mathrm{pH}-c_{0})\big)\right\}.
$$

### 2.3 Nitrification Rate
The daily nitrification flux (mass per area per day) is
$$
R_{\mathrm{nit}}(t) = \eta\;B(t)\;F_{w}\big(w(t)\big)\;F_{T}\big(T(t)\big)\;F_{\mathrm{pH}}\big(\mathrm{pH}\big),
$$
where \(\eta\) is a scaling factor (see below). In implementation:
$$
R_{\mathrm{nit}}(t) = \eta\;\big(k_{1}\,H_{\mathrm{dec}}(t)+k_{\max}\,N_{\mathrm{NH4}}(t)\big)\;F_{w}\,F_{T}\,F_{\mathrm{pH}}.
$$

---

## Parameters and Defaults
- **Kinetics:** \(k_{1}\) (humus-derived), \(k_{\max}\) (NH\(_4^+\)-derived) — soil-specific.  
- **WFPS response:** \(c_{0}=0.5\), \(c_{1}=0.0\), \(c_{2}=1.5\), \(c_{3}=4.5\).  
- **Temperature response:** \(c_{1}=-5.0\), \(c_{2}=7.0\), \(c_{3}=4.5\).  
- **pH response:** \(c_{0}=5.0\), \(c_{1}=0.56\), \(c_{2}=1.0\), \(c_{3}=0.45\).  
- **Scale factor:** \(\eta=1000\) (unit/scale alignment with model outputs).  
- **Soil heterogeneity:** soil-specific parameter sets (e.g., \(k_{1},k_{\max}\), porosity, depth, pH) selected by soil ID.

## Variable and Function Glossary
- \(N_{\mathrm{NH4}}(t)\): ammonium pool.  
- \(H_{\mathrm{dec}}(t)\): daily N released from humus decomposition.  
- \(w(t)\): water-filled pore space (WFPS).  
- \(T(t)\): soil temperature (°C).  
- \(\mathrm{pH}\): soil acidity/alkalinity.  
- \(F_{w}, F_{T}, F_{\mathrm{pH}}\): environmental limitation functions.  
- \(R_{\mathrm{nit}}(t)\): nitrification rate (flux).  
- \(\eta\): scale factor.


# Process-Based Nitrification Module (Parton–type Formulation)

## 1. Overview
We implement a process-based nitrification module following Parton–type response functions. The daily nitrification rate combines a **humus-derived source** from decomposition and an **NH₄⁺-driven potential**, **modulated multiplicatively** by environmental scalars for soil moisture (WFPS), temperature, and pH. Soil- and layer-specific parameters (e.g., porosity, depth, pH, bulk density) represent heterogeneity across soils. Unit conversions are applied between mass- and area-based forms to maintain consistency with the rest of the model.

## 2. Governing Formulation

### 2.1 Sources (humus- and NH₄⁺-driven)
Let \(H_{\mathrm{dec}}(t)\) be the daily nitrogen released from humus decomposition (mass units), and \(N_{\mathrm{NH4}}(t)\) the ammonium pool. A fraction of decomposed humus contributes directly to nitrification after conversion to concentration units; an NH₄⁺-driven term provides the potential substrate for nitrification:
\[
S_{h}(t) \;=\; f_{h}\,C_{\mathrm{humus,dec}}(t), 
\qquad
S_{n}(t) \;=\; k_{\max}\,C_{\mathrm{NH4}}(t),
\]
where \(f_{h}\approx 0.2\) (humus-to-nitrification fraction), \(k_{\max}\) is the maximum nitrification rate constant, and \(C_{\mathrm{humus,dec}},\,C_{\mathrm{NH4}}\) are concentrations (e.g., mg g\(^{-1}\)) obtained from pool masses using bulk density and layer depth.

### 2.2 Environmental Limitation Functions
**(a) Water limitation (WFPS):**
\[
F_{w}(w)=
\left(\frac{w-b}{a-b}\right)^{\;d\,\tfrac{(b-a)}{(a-c)}} 
\left(\frac{w-c}{a-c}\right)^{d},
\]
where \(w\) is water-filled pore space, and \(a,b,c,d\) are shape parameters.

**(b) Temperature limitation:**
\[
F_{T}(T) = a + b\,\exp(cT),
\]
with coefficients \((a,b,c)\) controlling baseline, amplitude, and sensitivity.

**(c) pH limitation:**
\[
F_{\mathrm{pH}}(\mathrm{pH}) = a + \frac{1}{\pi}\arctan(\pi b (c+\mathrm{pH})),
\]
with parameters \((a,b,c)\) shaping the curve.

### 2.3 Nitrification Rate
The daily nitrification flux (mass per area per day) is
\[
R_{\mathrm{nit}}(t) = \eta \Big(S_{h}(t)+S_{n}(t)\Big) F_{w}\big(w(t)\big) F_{T}\big(T(t)\big) F_{\mathrm{pH}}(\mathrm{pH}),
\]
where \(\eta\) is a scale/units conversion factor to reconcile concentration- to area-based flux.

---

## Parameters and Defaults
- **Kinetics:** \(f_{h}=0.2\), \(k_{\max}\) (soil-specific).  
- **WFPS response:** \(a=0.4, b=1.7, c=-0.007, d=3.22\).  
- **Temperature response:** \(a=-0.06, b=0.13, c=0.07\).  
- **pH response:** \(a=0.56, b=0.45, c=-5.0\).  
- **Bulk density:** default 1.52 g cm\(^{-3}\).  
- **Scale factor:** \(\eta=1000\) (to align with model units).  
- **Soil heterogeneity:** soil-specific \(k_{\max}\), porosity, depth, pH, humus decay rate, bulk density.

## Variable and Function Glossary
- \(N_{\mathrm{NH4}}(t)\): ammonium pool.  
- \(H_{\mathrm{dec}}(t)\): daily N released from humus decomposition.  
- \(C_{\mathrm{NH4}},C_{\mathrm{humus,dec}}\): concentrations (mg g\(^{-1}\)).  
- \(w(t)\): water-filled pore space (WFPS).  
- \(T(t)\): soil temperature (°C).  
- \(\mathrm{pH}\): soil acidity/alkalinity.  
- \(F_{w},F_{T},F_{\mathrm{pH}}\): environmental limitation functions.  
- \(R_{\mathrm{nit}}(t)\): nitrification flux (mass per area per day).  
- \(\eta\): scale/units conversion factor.

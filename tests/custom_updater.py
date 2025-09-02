import torch
import torch.nn as nn
from ecohydromodel import Updater

class MLPUpdater(Updater):
    """MLP-based updater that predicts nitrification from states and soil embedding."""

    def __init__(self, hidden_dim=64, num_layers=4, update_mode='parallel',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Build soil embedding, layer norm, and MLP head."""
        super().__init__({}, update_mode, device)
        self.soil_emb = nn.Embedding(225, 8)            # soil_id -> 8D embedding
        in_dim = 4 + 8                                   # features + soil_emb
        self.norm = nn.LayerNorm(in_dim)

        layers, d = [], in_dim
        for _ in range(num_layers - 1):                  # hidden blocks
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        layers += [nn.Linear(hidden_dim, 1)]             # output: nitrification rate
        self.net = nn.Sequential(*layers)

    def update(self, updated_state, state, hidden=None):
        """Predict nitrification and write back to updated_state."""
        soil_feat = self.soil_emb(state['soil'].long())
        # features: [nh4, humus, storage, TEMP_SOIL] + soil_emb
        x = torch.stack([state['nh4'], state['humus'],
                         updated_state['storage'], updated_state['TEMP_SOIL']], dim=-1)
        x = torch.cat([x, soil_feat], dim=-1)
        x = self.norm(x)
        nitrification = self.net(x).squeeze(-1)                  # predicted fraction
        
        # Update states
        updated_state['nitrification'] = (nitrification+updated_state['nitrification'])/2
        return updated_state, hidden


class BucketUpdater(Updater):
    """Simple bucket water balance updater with runoff, ET, and storage."""

    def __init__(self, params=None, update_mode='parallel',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize with default parameters and iteration vars."""
        defaults = {'k_runoff': 0.2, 'k_et': 0.5, 'max_storage': 120}
        final = {**defaults, **(params or {})}
        self.set_iter_vars(['outflow'])
        super().__init__(final, update_mode, device)

    def update(self, updated_state, state, hidden=None):
        """Update storage, outflow, and ET given rainfall and inflow."""
        RAIN, TEMP = updated_state['RAIN'], updated_state['TEMP']
        S = state['storage']
        Smax, k_run, k_et = self.max_storage, self.k_runoff, self.k_et

        # Evapotranspiration
        et = self.cal_et(TEMP, k_et=k_et)

        # Inflow from upstream
        Q_p = torch.where(RAIN > et, S * k_run, torch.zeros_like(S))
        inflow = self.aggregate(Q_p, weight=self.edge_attr['flow_ratio'])

        # Storage update
        W = S + RAIN + inflow
        ET_eff = torch.minimum(et, W)
        Q_eff = torch.minimum(Q_p, torch.clamp(W - ET_eff, min=0.0))
        S_star = S + RAIN + inflow - ET_eff - Q_eff

        # Overflow handling
        excess = torch.relu(S_star - Smax)
        S_next, Q_eff = S_star - excess, Q_eff + excess
        
        # Update states
        updated_state['storage'] = S_next
        updated_state['outflow'] = Q_eff
        updated_state['ET_eff'] = ET_eff
        return updated_state, hidden

    def cal_et(self, temp, k_et=1.0, daylength=12):
        """Compute potential ET with Hamon (1961)."""
        esat = 6.108 * torch.exp(17.27 * temp / (temp + 237.3))
        Pt = 216.7 * esat / (temp + 273.16)
        et = 0.1651 * k_et * daylength * Pt
        return torch.where(temp < 0, torch.zeros_like(et), et)

  
class NitriUpdater(Updater):
    """Nitrification process updater with Del Grosso or Parton equations."""

    def __init__(self, params=None, update_mode='parallel', equation='delgrosso',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize soil parameters and choose nitrification equation."""
        defaults = {
            'soil': {
                'Shallow_CN17': {'id': 217,'nitr_k1': 0.2, 'nitr_kmax': 0.2,'porosity': 0.453, 'depth': 265.0},
                'Shallow_CN12': {'id': 212,'nitr_k1': 0.2,'nitr_kmax': 0.1,'porosity': 0.453,'depth': 265.0},
                'Deep_CN12':    {'id': 112,'nitr_k1': 0.2,'nitr_kmax': 0.1,'porosity': 0.453,'depth': 725.0},
                'Medium_CN24':  {'id': 24, 'nitr_k1': 0.2,'nitr_kmax': 0.15,'porosity': 0.453,'depth': 277.5},
                'Deep_CN17':    {'id': 117,'nitr_k1': 0.2,'nitr_kmax': 0.20,'porosity': 0.453,'depth': 725.0},
                'Medium_CN12':  {'id': 12, 'nitr_k1': 0.2,'nitr_kmax': 0.1,'porosity': 0.453,'depth': 277.5},  
                'Medium_CN17':  {'id': 17, 'nitr_k1': 0.2,'nitr_kmax': 0.2,'porosity': 0.453,'depth': 277.5},   
                'Shallow_CN24': {'id': 224,'nitr_k1': 0.2,'nitr_kmax': 0.15,'porosity': 0.453,'depth': 265.0},
                'Deep_CN24':    {'id': 124,'nitr_k1': 0.2,'nitr_kmax': 0.15,'porosity': 0.453,'depth': 725.0},
            },
            'pH': 4.5,
            'humusNmaxDecay': 0.0002,
            'bulkDensity': 1.52,
            'psm_q': 0.015
        }
        final = {**defaults, **(params or {})}
        final = self.flatten_params(final, id_keys=['id'])
        self.equation = equation
        super().__init__(final, update_mode, device)

    def update(self, updated_state, state, hidden=None):
        """Compute nitrification amount and update state."""
        soil_id = state['soil']
        soil_param = self.lookup_params(soil_id, group='soil', index_by='id')

        # Prepare states
        moisture = updated_state['storage'] / (soil_param['porosity'] * soil_param['depth'])
        temp, humus, nh4 = updated_state['TEMP_SOIL'], state['humus'], state['nh4']

        # Parameters
        humusNmaxDecay = self.humusNmaxDecay
        psm_q, k1, kmax = self.psm_q, soil_param['nitr_k1'], soil_param['nitr_kmax']
        ph, bulkDensity = self.pH, self.bulkDensity
        soilDepth = soil_param['depth']

        # Humus decomposition -> NH4 supply
        humus_decomp = self.get_decomposition_amount(humus, humusNmaxDecay, moisture, temp)
        nh4_decomp = humus_decomp * (1 - psm_q)

        # Nitrification calculation
        if self.equation == 'delgrosso':
            nitrif = self.get_nitrification_amount_delgrosso(nh4_decomp, nh4, k1, kmax, moisture, temp, ph)
        elif self.equation == 'parton':
            nitrif = self.get_nitrification_amount_parton(nh4_decomp, nh4, k1, kmax,
                                                          moisture, temp, ph, bulkDensity, soilDepth)
        nitrif = torch.relu(nitrif)
        # Update states
        updated_state['nitrification'] = nitrif
        return updated_state, hidden

    def get_nitrification_amount_delgrosso(self, nh4_decomp, nh4, k1, kmax, moisture, temp, ph):
        """Del Grosso nitrification response to water, temp, and pH."""
        def water_effect(w, a=0.5, b=0.0, c=1.5, d=4.5):
            baseA = torch.clamp((w - b) / (a - b), min=0)
            baseB = torch.clamp((w - c) / (a - c), min=0)
            expA, expB = d * (b - a) / (a - c), d
            return baseA**expA * baseB**expB

        def temp_effect(t, a=-5.0, b=7.0, c=4.5):
            t1 = ((a - t) / (a - 25)).clamp(min=0)
            t2 = 1 - t1**c
            return t1**b * torch.exp(t2 * b / c)

        def ph_effect(ph, a=5.0, b=0.56, c=1.0, d=0.45):
            return (b + c / torch.pi * torch.atan(torch.pi * d * (ph - a))).clamp(max=1.0)

        limit = water_effect(moisture) * temp_effect(temp) * ph_effect(ph)
        base = k1 * nh4_decomp + kmax * nh4
        return base * limit

    def get_nitrification_amount_parton(self, nh4_decomp, nh4, k1, kmax, moisture, temp, ph, bulkDensity, soilThickness):
        """Parton nitrification response to water, temp, and pH."""
        def water_effect(w, a=0.4, b=1.7, c=-0.007, d=3.22):
            l = torch.clamp((w - b) / (a - b), min=0)
            r = torch.clamp((w - c) / (a - c), min=0)
            return l**(d * (b - a) / (a - c)) * r**d

        def temp_effect(t, a=-0.06, b=0.13, c=0.07):
            return a + b * torch.exp(c * t)

        def ph_effect(ph, a=0.56, b=0.45, c=-5.0):
            return a + torch.atan(torch.pi * b * (c + ph)) / torch.pi

        def to_mgg(amount, depth, bd): return (amount * 1e6) / (100*100*depth) / bd
        def to_gm2(amount, depth, bd): return (amount * bd * (100*100*depth)) / 1e6

        humus_mgg = to_mgg(nh4_decomp, soilThickness, bulkDensity)
        nh4_mgg = to_mgg(nh4, soilThickness, bulkDensity)
        nitrif_mgg = humus_mgg * k1 + kmax * nh4_mgg * water_effect(moisture) * temp_effect(temp) * ph_effect(ph)
        nitrif = to_gm2(nitrif_mgg, soilThickness, bulkDensity)
        return torch.where(temp < 0, torch.zeros_like(nitrif), nitrif)

    def get_decomposition_amount(self, nitrogen, max_decay, moisture, temp):
        """Decompose humus with moisture and temperature scalars."""
        def moist_scalar(m, lam=0.77, k=1.6):
            r = m / lam
            return (k / lam) * (r ** (k - 1)) * torch.exp(-(r ** k))

        def temp_scalar(t, qx=0.22658, maxT=50.0, optT=30.0):
            t_clip = torch.minimum(t, torch.tensor(maxT, dtype=t.dtype, device=t.device))
            r = (maxT - t_clip) / (maxT - optT)
            return torch.exp(qx * (t_clip - optT)) * (r ** (qx * (maxT - optT)))

        return nitrogen * max_decay * moist_scalar(moisture) * temp_scalar(temp) * 0.55

    
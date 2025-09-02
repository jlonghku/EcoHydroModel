# Updater Specification

This document provides guidelines for creating **custom updater modules** within the EcoHydroModel framework. 


**Note**: A simple approach is to upload this **module specification** to an LLM (e.g., GPT, Claude, Gemini) together with equations or conceptual descriptions, and let the LLMs automatically generate the corresponding updater implementation.

---

## 1. Basic Structure
All custom modules must inherit from the `Updater` base class and implement the `update` method:

```python
from ecohydromodel import Updater

class CustomUpdater(Updater):
    def __init__(self, params=None, update_mode='parallel', device='cuda'):
        super().__init__(params or {}, update_mode, device)

    def update(self, updated_state, state, hidden=None):
        # Define custom update logic
        updated_state['var'] = state['var'] * self.k
        return updated_state, hidden
```

---

## 2. Parameter Management

There are two ways to manage parameters in EcoHydroModel:

1. **Direct registration with `register_parameter`, `nn.Parameter`, `nn.ParameterDict`, `nn.ParameterList`**  
   You can explicitly define trainable parameters in the module constructor:  
   ```python
    self.register_parameter("k", nn.Parameter(torch.tensor(0.5)))
    self.k = nn.Parameter(torch.tensor(0.5))
    self.params = nn.ParameterDict({"k": nn.Parameter(torch.tensor(0.5))})
    self.params = nn.ParameterList([nn.Parameter(torch.tensor(0.5))])
   ```

2. **Passing parameter dictionaries to `super().__init__`**  
   You can pass a parameter dictionary to the `Updater` base class, and it will automatically be registered as trainable parameters. After registration, the parameters can be accessed directly as attributes, e.g. `self.k` or `self.max_storage`. 
   ```python
   defaults = {"k": 0.5, "max_storage": 100.0}
   super().__init__(defaults, update_mode="parallel", device=device)
   ```
    
    - If the parameter dictionary is **multi-level nested**, you can first flatten it with `self.flatten_params` before passing it in.
3. **ID-based Parameter Lookup & Broadcasting**     
   If you want to **dynamically look up parameters by ID** (e.g., soil types), you can use `self.lookup_params` inside the `update` method to rebuild **node-wise tensors** from an **ID tensor** at runtime.

Example:
```python
defaults = {
    "soil": {
        "typeA": {"id": 1, "k": 0.2},
        "typeB": {"id": 2, "k": 0.3}
    }
}
params = self.flatten_params(defaults, id_keys=["id"])
super().__init__(params, update_mode="parallel", device=device)

# Later in update():
soil_id=state["soil"]
soil_param = self.lookup_params(soil_id, group="soil", index_by="id")
```

---

## 3. Update Function

The `update(updated_state, state, hidden=None)` function defines how state variables evolve at each step. 

In practice, you only need to use the variables from `updated_state` (latest step) and `state` (previous step) to write your own process equations. (It is recommended to use `torch` operations for efficiency and compatibility with automatic differentiation.)

**Function arguments:**
- `updated_state`: Contains the **latest values** of all state variables at the current step, including forcings.  
- `state`: Contains the **previous time step values** of state variables, use for referencing past conditions.  
- `hidden`: A placeholder variable, intended for hidden states such as those used in deep learning modules (e.g., RNN or LSTM memory).  

EcoHydroModel supports several update modes that define how the `Updater` processes state evolution:

- **parallel**: Update all nodes simultaneously using the defined process equations.  
- **layer**: Update nodes layer by layer, following a topological order (e.g., in a DAG).  
- **converge**: Iteratively update variables until convergence criteria are satisfied.  
  - Requires calling `set_iter_vars(vars=[...], max_iters=N, tol=1e-6)` to define which variables should be iterated.  
- **max_depth**: Iteratively update for a fixed number of steps, simulating propagation along graph depth.  
  - Also requires `set_iter_vars` to specify which variables are updated at each step.  

---

## 4. Aggregate Function

The `aggregate(state, aggr='add', weight=None)` function collects neighbor values along graph edges.

- **state**: node features to be aggregated  
- **aggr**: aggregation method, options are `'add'`, `'mean'`, `'max'`  
- **weight**: optional edge weights (e.g., flow ratios)  

**Example**
```python
# Weighted inflow using flow ratios
inflow = self.aggregate(Q_p, aggr='add', weight=self.edge_attr['flow_ratio'])

# Simple neighbor mean
neighbor_mean = self.aggregate(x, aggr='mean')
```



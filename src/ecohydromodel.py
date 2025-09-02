import os,copy
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize
import matplotlib.cm as cm

import torch
import torch.nn as nn

import networkx as nx
from torch_geometric.data import  Data
from torch_geometric.utils import to_undirected,to_networkx
from torch_geometric.nn import graclus
from torch_geometric.nn.pool.pool import pool_edge, pool_pos
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_scatter import scatter

from pysheds.grid import Grid
from pyproj import Proj
from tensordict import TensorDict

import torch
import torch.nn as nn
from torch_scatter import scatter
import networkx as nx
from collections import defaultdict
import numpy as np


class Updater(nn.Module):
    """Base updater with graph utilities and multi-mode update logic."""

    def __init__(self, params=None, update_mode=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize learnable params and update mode."""
        super().__init__()

        params = params or {}

        for k, v in params.items():
            self.register_parameter(
                k, nn.Parameter(torch.tensor(v, dtype=torch.float32, device=device))
            )

        self.update_mode = update_mode


    # ---------- Hooks to be implemented by subclass ----------
    def update(self, updated_state, state, hidden):
        """One-step update (must be implemented in subclass)."""
        raise NotImplementedError("Subclasses must implement this method")


    # ---------- Forward driver for different update modes ----------
    def forward(self, updated_state, state, hidden):
        """Run updater according to update_mode."""
        if self.update_mode == 'layer':
            for nodes in self.layer_nodes.values():
                out, hidden = self.update(updated_state, state, hidden)
                updated_state[:, nodes] = out[:, nodes]

        elif self.update_mode == 'parallel':
            updated_state, hidden = self.update(updated_state, state, hidden)

        elif self.update_mode == 'converge':
            updated_state, hidden = self.update(updated_state, state, hidden)
            for _ in range(self.max_iters):
                out, h1 = self.update(updated_state.clone(), state, hidden)
                diffs = [(out[v] - updated_state[v]).abs().max() for v in self.iter_vars]
                if torch.stack(diffs).max() < self.tol:
                    updated_state, hidden = out, h1
                    break
                for v in self.iter_vars:
                    updated_state[v] = out[v]
                hidden = h1

        elif self.update_mode == 'max_depth':
            updated_state, hidden = self.update(updated_state, state, hidden)
            for i in range(self.graph_depth):
                out, h1 = self.update(updated_state.clone(), state, hidden)
                if i == self.graph_depth - 1:
                    updated_state, hidden = out, h1
                    break
                for v in self.iter_vars:
                    updated_state[v] = out[v]
                hidden = h1
        else:
            raise ValueError(f"Unsupported update_mode: {self.update_mode}")

        return updated_state, hidden
    

    # ---------- Param utilities ----------
    def flatten_params(self, nested_dict, id_keys=None):
        """Flatten nested param dicts into flat arrays/maps."""
        flat = {}
        id_keys = set(id_keys or [])
        for group, group_dict in nested_dict.items():
            if isinstance(group_dict, dict) and all(isinstance(v, dict) for v in group_dict.values()):
                grouped = defaultdict(list)
                for params in group_dict.values():
                    for k, v in params.items():
                        grouped[k].append(v)
                for k, values in grouped.items():
                    key = f"{group}_{k}"
                    arr = np.array(values)
                    if k in id_keys:
                        id_map = np.full(arr.max() + 1, -1, dtype=np.int32)
                        id_map[arr] = np.arange(len(arr))
                        flat[key] = id_map
                    else:
                        flat[key] = arr
            else:
                flat[group] = group_dict
        return flat

    def lookup_params(self, id_tensor, group, param_names=None, index_by=None):
        """Look up parameters for given ids from flattened dict."""
        params=dict(self.named_parameters())
        id = id_tensor.long()
        index_key = f'{group}_{index_by}' if index_by else f'{group}_id'
        index = params[index_key].long()[id]
        if param_names is None:
            prefix = f'{group}_'
            param_names = [k[len(prefix):] for k in params if k.startswith(prefix) and k != index_key]
        return {name: params[f'{group}_{name}'][index] for name in param_names}


    # ---------- Public utilities ----------
    def set_graph(self, edge_index, edge_attr):
        """Attach graph and prepare according to update_mode."""
        self.edge_index, self.edge_attr = edge_index, edge_attr
        if self.update_mode == 'layer':
            self.layer_nodes = self._compute_node_layers(edge_index)
        elif self.update_mode == 'max_depth':
            self.graph_depth = self._compute_graph_depth(edge_index)
        elif self.update_mode == 'parallel':
            pass
        elif self.update_mode == 'converge':
            if not self.iter_vars:
                raise ValueError("iter_vars must be set for 'converge' mode")
        else:
            raise ValueError(f"Unsupported update_mode: {self.update_mode}")
        
    def aggregate(self, state, aggr='add', weight=None):
        """Aggregate neighbor features along edges."""
        state = state.transpose(0, 1)
        src, dst = self.edge_index
        msg = state[src]
        if weight is not None:
            msg = msg * weight.unsqueeze(-1)
        out = scatter(msg, dst, dim=0, reduce=aggr, dim_size=state.size(0))
        return out.transpose(0, 1)

    def set_iter_vars(self, vars=None, max_iters=50, tol=1e-6):
        """Set variables for iterative convergence mode."""
        self.iter_vars = vars
        self.max_iters = max_iters
        self.tol = tol
    
    def set_ref_inputs(self, ref_inputs):
        """Store optional reference inputs."""
        self.ref_inputs = ref_inputs

    def set_ref_outputs(self, ref_outputs):
        """Store optional reference outputs."""
        self.ref_outputs = ref_outputs
    

    # ---------- Private helpers ----------
    def _build_graph(self, edge_index):
        """Build a directed graph from edge_index."""
        self.G = nx.DiGraph()
        self.G.add_edges_from(edge_index.t().tolist())
        return self.G

    def _compute_graph_depth(self, edge_index):
        """Compute longest path length as graph depth."""
        G = self._build_graph(edge_index)
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Graph must be a DAG to compute depth.")
        self.graph_depth = len(nx.dag_longest_path(G))
        return self.graph_depth

    def _compute_node_layers(self, edge_index):
        """Topologically group nodes into layers."""
        G = self._build_graph(edge_index)
        device = edge_index.device
        layer_nodes = {}
        visited = set()
        layer_idx = 0
        current = [n for n in G.nodes if G.in_degree(n) == 0]
        visited.update(current)

        while current:
            layer_nodes[layer_idx] = torch.tensor(current, device=device, dtype=torch.long)
            next_layer = []
            for n in current:
                for succ in G.successors(n):
                    if succ not in visited and all(p in visited for p in G.predecessors(succ)):
                        next_layer.append(succ)
                        visited.add(succ)
            current = next_layer
            layer_idx += 1

        self.layer_nodes = layer_nodes
        return layer_nodes


class Trainer:
    """Minimal training helper with checkpointing, clamping, and best-param tracking."""

    def __init__(self, model):
        """Store model reference."""
        self.model = model


    def train(self, data, epochs=100, lr=1e-3, chunk_size=None, ref_model=None, target_keys=None,
              trainable_params=None, not_trainable_params=None, param_ranges=None, loss_fn=nn.MSELoss(),
              print_freq=1, save_freq=10, save_path=None, load_path=None):
        """Train the model against reference or teacher outputs and return final params."""
        # Select trainable subset
        for n, p in self.model.named_parameters():
            p.requires_grad = (
                (trainable_params is None or any(n.startswith(k) for k in trainable_params)) and 
                (not_trainable_params is None or not any(n.startswith(k) for k in not_trainable_params))
            )

        optimizer = torch.optim.Adam((p for p in self.model.parameters() if p.requires_grad), lr=lr)
        self.model.train()

        # Optional data chunking
        if chunk_size is not None:
            data = data.chunk_data(chunk_size)

        # Build training targets
        with torch.no_grad():
            target = data.ref if ref_model is None else ref_model.eval()(data)

        # Optionally warm-start from checkpoint
        start_epoch, best_loss = 0, float('inf')
        if load_path and os.path.exists(load_path):
            start_epoch, best_loss = self.load_checkpoint(optimizer, load_path)

        best_params = None

        for epoch in range(start_epoch, epochs):
            optimizer.zero_grad()

            preds = self.model(data)
            keys = target_keys if target_keys is not None else (preds.keys() & target.keys())
            # loss = torch.stack([loss_fn(preds[k], target[k]) for k in keys]).mean()
            # compute scale-invariant multi-target loss
            eps, losses = 1e-8, []
            for k in keys:
                y, p = target[k], preds[k]
                mask = torch.isfinite(y) & torch.isfinite(p)   # skip invalid entries
                if mask.any():
                    s = y[mask].std().detach().clamp_min(eps)  # normalize by target std
                    losses.append(loss_fn(p[mask] / s, y[mask] / s))
            loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=next(self.model.parameters()).device)

            loss.backward()
            optimizer.step()

            # Clamp params to ranges if provided
            if param_ranges:
                self.clamp_parameters(param_ranges)

            # Track best
            best_loss, snap = self.update_best_params(best_loss, loss.item())
            if snap is not None:
                best_params = snap

            # Logging
            if (epoch + 1) % print_freq == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, Best: {best_loss:.6f}")

            # Periodic checkpoint
            if save_path and ((epoch + 1) % save_freq == 0 or epoch == epochs - 1):
                self.save_checkpoint(optimizer, save_path, epoch + 1, best_loss)

        # Restore best snapshot if available
        if best_params is not None:
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        p.copy_(best_params[n])

        # Final save
        if save_path:
            self.save_checkpoint(optimizer, save_path, epochs, best_loss)

        # Return a flat dict of final trainable params (by base name)
        return {n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad}


    def load_checkpoint(self, optimizer, path):
        """Load model/optimizer states and return (epoch, best_loss)."""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', 0), checkpoint.get('best_loss', float('inf'))


    def save_checkpoint(self, optimizer, path, epoch, best_loss):
        """Save model/optimizer states with epoch and best_loss."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, path)


    def clamp_parameters(self, param_ranges):
        """Clamp trainable parameters by provided ranges."""
        with torch.no_grad():
            for k, (mn, mx) in param_ranges.items():
                p = dict(self.model.named_parameters())[k]
                if p.requires_grad:
                    p.clamp_(mn, mx)


    def update_best_params(self, best_loss, current_loss):
        """Return (new_best_loss, snapshot) if improved, else (best_loss, None)."""
        if current_loss < best_loss:
            snap = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
            return current_loss, snap
        return best_loss, None


class Visualizer:
    """Visualization helpers for spatial-temporal outputs and metrics."""

    @staticmethod
    def plot(output, pos, name="Output", edge_index=None,
             plot_type='grid', dynamic_cbar=False, cbar_range=None):
        """Plot spatial output (grid or graph) with optional time slider."""
        if torch.is_tensor(output):
            output = output.detach().cpu().numpy()

        if plot_type == 'graph':
            pos = pos['pos_cluster'] if pos.get('cluster') is not None else pos['pos']
            pos = pos.detach().cpu().numpy()
        else:
            if pos.get('cluster') is not None:
                output = output[pos['cluster'].detach().cpu().numpy()]
            pos = pos['pos'].detach().cpu().numpy()

        vmin = cbar_range[0] if cbar_range else np.nanmin(output)
        vmax = cbar_range[1] if cbar_range else np.nanmax(output)

        fig, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.25)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = 'viridis'
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(mapper, ax=ax)
        cbar.set_label(name)
        cbar.set_ticks(np.linspace(vmin, vmax, 5))

        def draw_graph(t):
            ax.clear()
            values = output[:, t]
            if dynamic_cbar and not cbar_range:
                mapper.set_norm(Normalize(vmin=values.min(), vmax=values.max()))
                cbar.update_normal(mapper)

            if plot_type == 'graph':
                data = Data(edge_index=edge_index, num_nodes=output.shape[0], pos=pos)
                G = to_networkx(data)
                nx.draw(G, pos=pos, ax=ax, with_labels=False,
                        node_color=mapper.to_rgba(values), node_size=300, edge_color='gray')
            else:
                xu, yu = np.unique(pos[:, 0]), np.unique(pos[:, 1])
                grid = np.full((len(yu), len(xu)), np.nan)
                for i, (xv, yv) in enumerate(pos):
                    xi, yi = np.where(xu == xv)[0][0], np.where(yu == yv)[0][0]
                    grid[yi, xi] = values[i]
                ax.imshow(grid, cmap=cmap, norm=mapper.norm, origin='lower',
                          extent=[xu.min(), xu.max(), yu.min(), yu.max()])
                ax.set_xlabel("X")
                ax.set_ylabel("Y")

            ax.set_title(f"{name} at Time Step {t}")
            fig.canvas.draw_idle()

        draw_graph(0)
        if output.shape[1] > 1:
            slider = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]),
                            'Time Step', 0, output.shape[1] - 1,
                            valinit=0, valstep=1)
            slider.on_changed(lambda val: draw_graph(int(val)))
        plt.show()

    @staticmethod
    def plot_similarity(pred, obs, pos, metric='nse', plot_type='grid', cbar_range=[0, 1]):
        """Plot similarity metric (NSE, KGE, NSE+KGE) between prediction and observation."""
        m = Visualizer.compute_metric(pred, obs, metric)
        Visualizer.plot(m.unsqueeze(-1), pos=pos, plot_type=plot_type,
                        name=metric, cbar_range=cbar_range)

    @staticmethod
    def compute_metric(pred, obs, metric='nse'):
        """Compute NSE/KGE/combined metrics between prediction and observation."""
        pred, obs = pred.float(), obs.float()
        mean_obs = obs.mean(dim=1, keepdim=True)

        if metric == 'nse':
            return 1 - ((pred - obs) ** 2).sum(dim=1) / ((obs - mean_obs) ** 2).sum(dim=1).add(1e-8)

        elif metric == 'kge':
            mean_pred, std_pred = pred.mean(dim=1), pred.std(dim=1)
            mean_obs, std_obs = mean_obs.squeeze(1), obs.std(dim=1)
            r = ((pred - mean_pred[:, None]) * (obs - mean_obs[:, None])).mean(dim=1) / (std_pred * std_obs + 1e-8)
            beta, gamma = mean_pred / (mean_obs + 1e-8), std_pred / (std_obs + 1e-8)
            return 1 - ((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2).sqrt()

        elif metric == 'nse_kge':
            nse = 1 - ((pred - obs) ** 2).sum(dim=1) / ((obs - mean_obs) ** 2).sum(dim=1).add(1e-8)
            mean_pred, std_pred = pred.mean(dim=1), pred.std(dim=1)
            mean_obs, std_obs = mean_obs.squeeze(1), obs.std(dim=1)
            r = ((pred - mean_pred[:, None]) * (obs - mean_obs[:, None])).mean(dim=1) / (std_pred * std_obs + 1e-8)
            beta, gamma = mean_pred / (mean_obs + 1e-8), std_pred / (std_obs + 1e-8)
            kge = 1 - ((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2).sqrt()
            return (nse + kge) / 2

        else:
            raise ValueError(f"Unsupported metric: {metric}")

    @staticmethod
    def plot_timeseries(output, obs=None, node_idx=None, name="State"):
        """Plot time series for one or multiple nodes."""
        if isinstance(output, torch.Tensor):
            output = output.detach().cpu().numpy()
        if obs is not None and isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()

        node_idx = [node_idx] if isinstance(node_idx, int) else (node_idx or [0])
        plt.figure(figsize=(8, 4))
        for idx in node_idx:
            plt.plot(output[idx], label=f"Predicted {name} (Node {idx})", linewidth=2)
            if obs is not None:
                plt.plot(obs[idx], "--", label=f"Observed {name} (Node {idx})", linewidth=2)
        plt.xlabel("Time step")
        plt.ylabel(name)
        plt.title(f"Time series ({name})")
        plt.legend()
        plt.tight_layout()
        plt.show()


class DataManager:
    """Manage states, forcings, references, positions, and graph data."""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize containers on given device."""
        self.device = device
        self.mask = None
        self.edge_index = None
        self.edge_attr = TensorDict(device=self.device)
        self.pos = TensorDict(device=self.device)
        self.state = TensorDict(device=self.device)
        self.ref = TensorDict(device=self.device)
        self.forcing = TensorDict(device=self.device)

    def clone(self):
        """Return a deep copy of self."""
        return copy.deepcopy(self)

    def load_csv(self, csv_paths, target=None, mask=None, time_index=None):
        """Load csv into TensorDict and assign to target if provided."""
        results = TensorDict(device=self.device)
        mask = mask or self.mask
        if isinstance(time_index, (list, tuple)) and len(time_index) == 2:
            time_index = slice(*time_index)

        for name, path in csv_paths.items():
            data = pd.read_csv(path, header=None).to_numpy()
            out = data.reshape(-1, *mask.shape)[time_index, mask].T
            results[name] = torch.tensor(out, dtype=torch.float32, device=self.device)
        results.batch_size = results[name].shape

        if target == "forcing":
            self.forcing.update(results); self.forcing.batch_size = results[name].shape
        elif target == "state":
            self.state.update(results); self.state.batch_size = results[name].shape
        elif target == "ref":
            self.ref.update(results); self.ref.batch_size = results[name].shape
        else:
            return results
        return results

    def save_csv(self, td, prefix="output", mode="single", mask=None):
        """Save TensorDict or Tensor to csv files."""
        if td.dim() == 2:
            td = td.transpose(1, 0)
        if isinstance(td, torch.Tensor):
            data = {"tensor": td.cpu().numpy().reshape(-1)}
        else:
            data = {k: v.cpu().numpy().reshape(-1) for k, v in td.items()}
        df = pd.DataFrame(data)

        if mode == "single":
            df.to_csv(f"{prefix}_all.csv", index=False)
            print(f"Saved to {prefix}_all.csv")
        elif mode == "split":
            base_mask = mask or self.mask
            if base_mask is not None:
                for k, v in data.items():
                    n = len(v) // base_mask.sum()
                    new_mask = np.tile(base_mask, (n, 1))
                    result = np.full(new_mask.shape, np.nan)
                    result[new_mask == 1] = v
                    pd.DataFrame(result).to_csv(f"{prefix}_{k}.csv", index=False, header=False)
                    print(f"Saved to {prefix}_{k}.csv")
            else:
                for k, v in df.items():
                    v.to_csv(f"{prefix}_{k}.csv", index=False, header=False)
                    print(f"Saved to {prefix}_{k}.csv")

    def save_tensordict(self, td, name="tensordict.pt"):
        """Save TensorDict to file."""
        torch.save(td, name)

    def load_tensordict(self, path="tensordict.pt", target=None):
        """Load TensorDict and assign to state/forcing/ref if specified."""
        td = torch.load(path, weights_only=False)
        if target == 'state': self.state = td
        elif target == 'forcing': self.forcing = td
        elif target == 'ref': self.ref = td
        return td

    def coarse(self, iters=1, reduce_dict=None, max_nodes=0):
        """Coarsen graph using graclus pooling until reaching max_nodes."""
        clone = self.clone()
        for td in [clone.state, clone.forcing, clone.ref, clone.edge_attr, clone.pos]:
            td.batch_size = []

        cluster_all = torch.arange(next(iter(clone.state.values())).shape[0], device=self.device)
        iters = int(1e8) if max_nodes > 0 else iters
        for _ in range(iters):
            cluster = graclus(to_undirected(clone.edge_index))
            cluster, perm = consecutive_cluster(cluster)
            for k, v in clone.state.items():
                clone.state[k] = clone.scatter(v, cluster, dim=0, reduce=reduce_dict.get(k, 'mean') if reduce_dict else 'mean')
            for k, v in clone.forcing.items():
                clone.forcing[k] = clone.scatter(v, cluster, dim=0, reduce=reduce_dict.get(k, 'mean') if reduce_dict else 'mean')
            for k, v in clone.ref.items():
                clone.ref[k] = clone.scatter(v, cluster, dim=0, reduce=reduce_dict.get(k, 'mean') if reduce_dict else 'mean')
            for k, v in clone.edge_attr.items():
                index, clone.edge_attr[k] = pool_edge(cluster, clone.edge_index, v, reduce=reduce_dict.get(k, 'mean') if reduce_dict else 'mean')
            clone.edge_index = index
            clone.pos['pos_cluster'] = pool_pos(cluster, clone.pos['pos_cluster'])
            cluster_all = cluster[cluster_all]
            if next(iter(clone.state.values())).shape[0] <= max_nodes:
                break

        clone.pos['cluster'] = cluster_all
        clone.state.batch_size = next(iter(clone.state.values())).shape
        clone.forcing.batch_size = next(iter(clone.forcing.values())).shape
        clone.ref.batch_size = next(iter(clone.ref.values())).shape
        clone.edge_attr.batch_size = next(iter(clone.edge_attr.values())).shape
        return clone

    def scatter(self, x, index, dim=0, dim_size=None, reduce="sum"):
        """Scatter with sum/mean/max or custom mode aggregation."""
        reduce = reduce.lower()
        if reduce != "mode":
            return scatter(x, index, dim=dim, dim_size=dim_size, reduce=reduce)

        if dim < 0: dim += x.ndim
        G = int(index.max().item()) + 1 if dim_size is None else int(dim_size)

        x_perm = x.movedim(dim, 0)
        N, *rest = x_perm.shape
        M = int(torch.tensor(rest).prod()) if rest else 1
        x2 = x_perm.reshape(N, -1)
        idx = index.reshape(N).long()

        uniq, inv = torch.unique(x2, return_inverse=True)
        inv = inv.view(N, -1)
        K = uniq.numel()

        counts = torch.zeros(G * K, M, device=x.device, dtype=torch.long)
        key_g = idx.unsqueeze(1).expand_as(inv)
        idx2d = key_g * K + inv
        counts.scatter_add_(0, idx2d, torch.ones_like(inv, dtype=torch.long))
        counts = counts.view(G, K, M)

        mode_id = counts.argmax(dim=1)
        out = uniq[mode_id].view(G, *rest).movedim(0, dim)

        group_sizes = torch.zeros(G, device=x.device, dtype=torch.long)
        group_sizes.scatter_add_(0, idx, torch.ones_like(idx))
        empty = group_sizes == 0
        if empty.any():
            out.index_copy_(dim, empty.nonzero(as_tuple=True)[0], torch.zeros_like(out.select(dim, 0)))
        return out

    def chunk_data(self, chunk_size):
        """Split time dimension into chunks for sequence training."""
        clone = self.clone()
        N, T = clone.forcing.shape
        num_chunks = T // chunk_size
        clone.forcing = clone.forcing[:, :num_chunks * chunk_size].reshape(N, num_chunks, chunk_size).permute(1, 0, 2)
        clone.ref = clone.ref[:, :num_chunks * chunk_size].reshape(N, num_chunks, chunk_size).permute(1, 0, 2)
        state = clone.state.clone()
        clone.state = TensorDict({}, batch_size=[num_chunks, N], device=self.device)
        clone.state[0] = state
        for i in range(num_chunks - 1):
            state.update(clone.ref[i, :, -1])
            clone.state[i + 1] = state
        return clone

    def load_dem(self, input_asc, outx, outy, xytype='index', crs="EPSG:4326",
                 extra_asc=None, index_asc=None):
        """Load DEM from ascii, extract catchment, build graph and features."""
        features = TensorDict(device=self.device)
        edge_attr = TensorDict(device=self.device)

        grid = Grid.from_ascii(input_asc, crs=Proj(crs))
        dem = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(grid.read_ascii(input_asc))))

        fdir = grid.flowdir(dem)
        mask = grid.catchment(x=outx, y=outy, fdir=fdir, xytype=xytype) == 1
        cells = np.argwhere(mask)
        if cells.size == 0:
            raise ValueError("No valid catchment found at outlet.")

        rows, cols = cells[:, 0], cells[:, 1]
        xs, ys = grid.affine * (cols + 0.5, rows + 0.5)
        pos = torch.tensor(np.stack([xs, ys], 1), dtype=torch.float32, device=self.device)

        cell_idx = {tuple(cell): i for i, cell in enumerate(cells)}
        d_row, d_col = np.array([-1,-1,0,1,1,1,0,-1]), np.array([0,1,1,1,0,-1,-1,-1])
        edges, flow_ratio, slopes = [], [], []
        cellsize_x, cellsize_y = grid.affine.a, -grid.affine.e

        for idx, (r, c) in enumerate(cells):
            curr_elev = dem[r, c]
            neighbors = [(idx, cell_idx[(r+dr, c+dc)], curr_elev - dem[r+dr, c+dc],
                          np.sqrt((dr*cellsize_y)**2 + (dc*cellsize_x)**2))
                        for dr, dc in zip(d_row, d_col)
                        if 0 <= r+dr < dem.shape[0] and 0 <= c+dc < dem.shape[1]
                        and mask[r+dr, c+dc] and dem[r+dr, c+dc] < curr_elev]
            slopes.append(sum(d/s for _,_,d,s in neighbors))
            t = sum(d for *_, d,_ in neighbors)
            for src, dst, d,_ in neighbors:
                edges.append([src, dst])
                flow_ratio.append(d / t if t > 0 else 0.0)

        edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()
        edge_attr['flow_ratio'] = torch.tensor(flow_ratio, dtype=torch.float32, device=self.device)

        features['DEM'] = torch.tensor(dem[mask], dtype=torch.float32, device=self.device)
        features['total_slope'] = torch.tensor(slopes, dtype=torch.float32, device=self.device)

        if extra_asc:
            for name, path in extra_asc.items():
                data = grid.read_ascii(path, crs=Proj(crs))
                features[name] = torch.tensor(data[mask], dtype=torch.float32, device=self.device)
        if index_asc:
            for name, path in index_asc.items():
                data = grid.read_ascii(path, crs=Proj(crs))
                features[name] = torch.tensor(data[mask], dtype=torch.long, device=self.device)

        self.edge_index = edge_index
        self.pos['pos'] = pos
        self.pos['pos_cluster'] = pos.clone()
        self.mask = mask
        self.state.update(features)
        self.state.batch_size = features['DEM'].shape
        self.edge_attr.update(edge_attr)
        return features, edge_index, edge_attr, pos

   
class EcoHydroModel(nn.Module):
    """Wrapper that runs a sequence of Updater modules over time."""

    def __init__(self, updater_modules, order=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Register updaters and execution order."""
        super().__init__()
        self.device = device
        self.order = order or list(updater_modules.keys())

        # Register each updater directly
        for name, mod in updater_modules.items():
            self.add_module(name, mod)

    def clone(self):
        return copy.deepcopy(self)

    def forward(self, data):
        """Roll the system forward through time using chained updaters."""
        data = data.clone()

        # Handle batch dimension
        has_batch = data.state.ndim == 2
        state, forcing, ref = data.state, data.forcing, data.ref
        if not has_batch:
            state = state.unsqueeze(0)
            forcing = forcing.unsqueeze(0)
            ref = ref.unsqueeze(0)

        # Attach graph info to each updater
        for name in self.order:
            getattr(self, name).set_graph(data.edge_index, data.edge_attr)

        state_all = TensorDict({}, batch_size=forcing.shape, device=self.device)

        hidden = None
        T = forcing.shape[-1]
        for t in range(T):
            updated_state = forcing[..., t].clone()
            state.lock_()

            # Apply each updater in sequence
            for name in self.order:
                upd = getattr(self, name)

                # Override inputs/outputs with reference if specified
                for key in getattr(upd, "ref_inputs", []) or []:
                    updated_state[key] = ref[key][..., t]
                updated_state, hidden = upd(updated_state, state, hidden)
                for key in getattr(upd, "ref_outputs", []) or []:
                    updated_state[key] = ref[key][..., t]

            state.unlock_()
            state.update(updated_state)
            state_all[..., t] = state

        return state_all if has_batch else state_all.squeeze(0)




  



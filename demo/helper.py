# Basic Imports
import torch
import numpy as np
from matplotlib import pyplot as plt

import vector
import tree
from IPython.display import display, Latex
from io import StringIO 
import sys        

from spanet import JetReconstructionModel, Options
from spanet.test import evaluate_predictions, display_table, display_latex_table


from torch import Tensor, nn
from typing import Optional
import types
    
import seaborn as sb
sb.set_theme("notebook", "whitegrid", font_scale=1.5, rc={"text.usetex": True, "figure.figsize": (9, 5)})

def assignment_index(data, assignments, mask):
    dummy_index = torch.arange(mask.sum())[:, None]
    
    return data[mask][dummy_index, assignments[mask]]

def invariant_mass(vectors):
    log_mass, log_pt, eta, sin_phi, cos_phi, *_ = vectors.permute(2, 0, 1).cpu().numpy().astype(np.float64)
    
    mass = np.expm1(log_mass)
    pt = np.expm1(log_pt)
    
    px = pt * cos_phi
    py = pt * sin_phi
    pz = pt * np.sinh(eta)
    
    E = np.sqrt(mass ** 2 + px ** 2 + py ** 2 + pz ** 2)
    
    p2 = px.sum(1) ** 2 + py.sum(1) ** 2 + pz.sum(1) ** 2
    
    return np.nan_to_num(np.sqrt(E.sum(1) ** 2 - p2))

def plot_distribution(dist, line = 0, label=None, range=None):
    if isinstance(dist, torch.Tensor):
        dist = dist.detach().cpu().numpy()
        
    if range is None:
        xmin = np.percentile(dist, 0.5)
        xmax = np.percentile(dist, 99.5)
    else:
        xmin, xmax = range
        
    plt.hist(
        dist,
        bins=128,
        range=(xmin, xmax),
        density=True,
        histtype="step" if line else "bar",
        linewidth = line if line else 1,
        label=label
    )
    plt.tight_layout()
    
def batch_to_gpu(batch):
    return tree.map_structure(lambda x: x.cuda(), batch)

def plot_1d_assignment_distribution(dist, num_vectors):
    dist = dist[0, :num_vectors].detach().cpu().numpy()
    x = np.arange(num_vectors)
    
    plt.bar(x, dist)
    plt.xlabel("Jet Index")
    plt.ylabel("Assignment Probability")
    
def plot_2d_assignment_distribution(dist, num_vectors):
    dist = dist[0, :num_vectors, :num_vectors].detach().cpu().numpy()
    
    plt.imshow(dist, vmin=0.0, vmax=0.5)
    
    for i in range(num_vectors):
        for j in range(num_vectors):
            val = dist[i, j]
            text = plt.text(j, i, f"{dist[i, j]:.2f}", ha="center", va="center", color="w" if val < 0.25 else "b")
        
    # plt.xlabel("$q_1$ Jet Index")
    # plt.ylabel("$q_2$ Jet Index")
    plt.xticks([]);
    plt.yticks([]);
    plt.colorbar()
    
def display_loss_table(symmetric_losses):
    from rich import get_console
    from rich.table import Table

    console = get_console()

    table = Table(title="Losses", header_style="bold magenta")
    table.add_column("Particle", justify="left")
    table.add_column("Assignment Loss", justify="left")
    table.add_column("Detection Loss", justify="left")

    losses = symmetric_losses.mean(-1).detach().cpu().numpy()
    names = ["Leptonic Top", "Hadronic Top"]
    for name, row in zip(names, losses):
        table.add_row(name, *map(str, row))

    console.print(table)
    
def efficiency(predictions, targets, masks):
    return (predictions == targets).all(1)[masks].mean()

def event_efficiency(predictions, targets, masks):
    predictions = np.concatenate(predictions, axis=1)
    targets = np.concatenate(targets, axis=1)

    masks = np.all(np.stack(masks), axis=0)
    
    return efficiency(predictions, targets, masks)

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout
        
def display_results_table(model, evaluation):
    predictions = list(evaluation.assignments.values())
    targets = [assignment[0].cpu().numpy() for assignment in model.testing_dataset.assignments.values()]
    masks = [assignment[1].cpu().numpy() for assignment in model.testing_dataset.assignments.values()]
    
    results, jet_limits, clusters = evaluate_predictions(
        predictions, 
        model.testing_dataset.num_vectors.cpu().numpy(), 
        targets, 
        masks, 
        model.options.event_info_file, 
        lines=2
    )

    with Capturing() as output:
        display_latex_table(results, jet_limits, clusters)

    plt.text(0.0, 1.0, " ".join(output), fontsize=14);
    plt.axis('off');
    
    
def roc_curve(target, pred, name):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(target, pred)
    plt.plot(fpr, tpr, label=f"{name} AUC: {auc(fpr, tpr):.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

def plot_input(input):
    masked_features = input[:].data[input[:].mask].numpy()
    num_features = len(input.input_features)
    fig, axes = plt.subplots(ncols=num_features, figsize=(num_features * 5, 4))
    for i, feature in enumerate(input.input_features):
        axes[i].hist(masked_features[:, i], bins=128, density=True)
        axes[i].set_xlabel(feature.name)
        # axes[i].set_ylabel("Density")
    plt.tight_layout()

def hook_attention(attention_layers):
    def forward(
        self, 
        query: Tensor, 
        key: Tensor, 
        value: Tensor, 
        key_padding_mask: Optional[Tensor] = None, 
        need_weights: bool = True, 
        attn_mask: Optional[Tensor] = None, 
        average_attn_weights: bool = True
    ):
        return nn.MultiheadAttention.forward(self, query, key, value, key_padding_mask, True, attn_mask, average_attn_weights)
    
    for attention_layer in attention_layers:
        attention_layer.forward = types.MethodType(forward, attention_layer)
        
    attention_outputs = {module: None for module in attention_layers}
    def attention_hook(module, input, output):
        attention_outputs[module] = output[1]
        
    handles = [
        attention_layer.register_forward_hook(attention_hook)
        for attention_layer in attention_layers
        
    ]

    return attention_outputs, handles

def plot_attention(model, attention_outputs, example_validation_event):
    predictions = model(example_validation_event.sources)

    num_vectors = example_validation_event.sources[0].mask.sum().item()
    attention_matrices = [
        attention_output[0].cpu()
        for attention_output in attention_outputs.values()
    ]
    
    good_indices = list(range(num_vectors + 1)) + [attention_matrices[0].shape[0] - 1]
    
    A = torch.eye(len(good_indices))
    for M in attention_matrices[-1:]:
        A = M[good_indices, :][:, good_indices] @ A


    labels = ["$\\emptyset$" for _ in range(A.shape[0])]
    labels[1] = "$\\mu_l$" if example_validation_event.sources[0].data[0, 0, -1].item() > 0.5 else "$e_l$"
    labels[example_validation_event.assignment_targets[0].indices[0][0].item() + 1] = "$b_l$"
    labels[example_validation_event.assignment_targets[1].indices[0][0].item() + 1] = "$b_h$"
    labels[example_validation_event.assignment_targets[1].indices[0][1].item() + 1] = "$q_h$"
    labels[example_validation_event.assignment_targets[1].indices[0][2].item() + 1] = "$q_h$"
    labels[-1] = "MET"
    labels[0] = "G"

    plt.imshow(A.numpy(), vmin=0.0)
    plt.xticks(np.arange(A.shape[0]), labels, minor=True);
    plt.yticks(np.arange(A.shape[0]), labels, minor=True);
    plt.xticks(np.arange(A.shape[0]) - 0.5, [])
    plt.yticks(np.arange(A.shape[0]) - 0.5, [])
    plt.colorbar()
    plt.tight_layout()

def make_labels(example_validation_event):
    num_vectors = example_validation_event.sources[0].mask.sum().item()
    
    labels = ["$\\emptyset$" for _ in range(num_vectors)]
    labels[0] = "$\\mu_l$" if example_validation_event.sources[0].data[0, 0, -1].item() > 0.5 else "$e_l$"
    labels[example_validation_event.assignment_targets[0].indices[0][0].item()] = "$b_l$"
    labels[example_validation_event.assignment_targets[1].indices[0][0].item()] = "$b_h$"
    labels[example_validation_event.assignment_targets[1].indices[0][1].item()] = "$q_h$"
    labels[example_validation_event.assignment_targets[1].indices[0][2].item()] = "$q_h$"

    return np.arange(num_vectors), labels
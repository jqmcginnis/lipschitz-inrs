import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from deel import torchlip

def get_activation_lipschitz(activation):
    """
    Infer Lipschitz constant from activation type.
    - None -> 1.0 (identity)
    - ReLU (F.relu, torch.relu, nn.ReLU) -> 1.0
    - SinusoidalActivation -> omega
    - GaussianActivation -> 1/(a*sqrt(e))
    """
    if activation is None:
        c = 1.0
    elif isinstance(activation, (nn.ReLU, torch.nn.modules.activation.ReLU)):
        c = 1.0
    elif activation in [F.relu, torch.relu]:
        c = 1.0
    elif isinstance(activation, GaussianActivation):
        c = 1 / (activation.a*math.sqrt(math.e))
    elif isinstance(activation, SinusoidalActivation):
        c = activation.omega.item()
    elif isinstance(activation,torchlip.GroupSort):
        c= 1.0
    elif isinstance(activation,torchlip.GroupSort2):
        c= 1.0
    elif isinstance(activation,torchlip.HouseHolder):
        c= 1.0   
    else:
        raise ValueError(f"Unsupported activation type: {type(activation)}")
    return c

def matrix_info(weight, activation, eps=1e-12):
    """Compute spectral norm, Frobenius norm, stable rank for a weight matrix."""
    act_lip = get_activation_lipschitz(activation)
    
    # Detach from computational graph for analysis
    with torch.no_grad():
        weight_detached = weight.detach()
        spectral_norm = torch.linalg.matrix_norm(weight_detached, ord=2)
        fro_norm = torch.norm(weight_detached, p='fro')
        stable_rank = (fro_norm ** 2) / (spectral_norm ** 2 + eps)
        spectral_condition_no = torch.linalg.cond(weight_detached, p=2)
    
    return {
        'linear_spectral_norm': spectral_norm.item(),
        'activation_spectral_norm': act_lip,
        'combined_spectral_norm': spectral_norm.item() * act_lip,
        'frobenius_norm': fro_norm.item(),
        'stable_rank': stable_rank.item(),
        'spectral_condition_no': spectral_condition_no.item(),
    }

class ActivatedLinear(nn.Module):
    """Linear layer with activation and spectral norm tracking."""
    def __init__(self, in_features, out_features, activation=None, layer_type="hidden", init_siren=False, alpha=1):
        super().__init__()

        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        self.layer_type = layer_type

        if layer_type != "wire_last":
            self.linear = nn.Linear(in_features, out_features)
        else:
            # in WIRE the last layer is complex-valued
            self.linear = nn.Linear(in_features, out_features, dtype=torch.cfloat)

        # edge case SIREN:
        # we do this for all sinusoidal activations
        if isinstance(activation, SinusoidalActivation):
            self.init_weights_siren(layer_type=self.layer_type,
                                    omega=self.activation.omega.item(),
                                    alpha=alpha)
            
        # for regular linear layers only if and when we use SIREN init (!)
        if init_siren and self.layer_type in ["last"]:
            self.init_weights_siren(layer_type=self.layer_type, omega=None)

    """ Weight init as in SIREN paper """
    def init_weights_siren(self, layer_type="first", omega=30, alpha=1):    
        with torch.no_grad():
            if layer_type == "first":
                # no division by omega here, as in the paper
                self.linear.weight.uniform_(-1*alpha / self.in_features, 
                                             1*alpha / self.in_features)      
            elif layer_type == "hidden":
                self.linear.weight.uniform_(-alpha*np.sqrt(6 / self.in_features) / omega, 
                                             alpha*np.sqrt(6 / self.in_features) / omega)
            elif layer_type == "last":
                # in SIREN the dim is hidden_features, but the in_features are hidden_features (!)
                # notice how we don't divide by omega here
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features), 
                                             np.sqrt(6 / self.in_features))
        
    def forward(self, x):
        if self.activation is None:
            if self.layer_type == "wire_last":
                return self.linear(x).real
            else:
                return self.linear(x)    
        else:           
            return self.activation(self.linear(x))

    def get_info(self):
        return matrix_info(self.linear.weight, self.activation)
    
class SinusoidalActivation(nn.Module):
    """Sinusoidal activation function as described in SIREN, Sitzmann et al. NeurIPS'20.
    
    Implements the Gaussian activation: sin(omega * x)
    """
    def __init__(self, omega=1.0):
        super().__init__()
        self.register_buffer('omega', torch.tensor(omega))

    def forward(self, x):
        return torch.sin(self.omega * x)
    
class GaussianActivation(nn.Module):
    """Gaussian activation function as described in Table 1, Beyond Periodicity: Towards a 
    Unifying Framework for Activations in Coordinate-MLPs, Ramasingh et al. ECCV'22.
    
    Implements the Gaussian activation: exp(-x²/(2a²))
    """
    def __init__(self, a=1.0):
        super().__init__()
        self.register_buffer('a', torch.tensor(a))

    def forward(self, x):
        return torch.exp(-0.5 * x**2 / (self.a**2))
    
class SirenMLP(nn.Module):
    """
    Sinusoidal-activated MLP as in SIREN, Sitzmann et al. NeurIPS'20.
    """
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=3, num_layers=4, omega=30.0, alpha=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha
        
        # omega could be a list for layer-specific values
        if isinstance(omega, (list, tuple)):
            if len(omega) != (num_layers-1):
                raise ValueError(f"Length of 'omega' list ({len(omega)}) must match num_layers -1 ({num_layers-1})")
            self.omega_values = omega
        else:
            self.omega_values = [omega] * (num_layers -1)

        # Build MLP with layer-specific 'a' values
        layers = [ActivatedLinear(input_dim, hidden_dim, SinusoidalActivation(omega=self.omega_values[0]), layer_type="first", init_siren=True, alpha=alpha)]
        for i in range(num_layers - 2):
            layers.append(ActivatedLinear(hidden_dim, hidden_dim, SinusoidalActivation(omega=self.omega_values[i+1]), layer_type="hidden", init_siren=True, alpha=alpha))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        # notice how we use init_siren=True here to trigger the special last-layer init, not the vanilla init for nn.Linear()
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None, layer_type="last", init_siren=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    def get_end_to_end_spectral_bound(self):
        total = torch.tensor(1.0, device=self.mlp[0].linear.weight.device)
        for info in self.get_layer_infos():
            total *= info['combined_spectral_norm']
        return total

    def get_detailed_matrix_info(self):
        infos = self.get_layer_infos()
        return {
            'layer_infos': infos,
            'end_to_end_spectral_bound': self.get_end_to_end_spectral_bound(),
            'omega_values': self.omega_values,
            'num_layers': len(infos)
        }

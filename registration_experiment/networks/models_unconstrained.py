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
    elif isinstance(activation, GaborComplexActivation):
        omega = float(activation.omega.item())
        sigma = float(activation.sigma.item())
        if sigma == 0.0:
            c = abs(omega)
        elif omega**2 >= 2.0 * sigma**2:
            c = abs(omega)
        else:
            term = math.sqrt(2.0) * sigma * math.exp((omega**2) / (4.0 * sigma**2) - 0.5)
            c = max(abs(omega), term)
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
    def __init__(self, in_features, out_features, activation=None, layer_type="hidden", init_siren=False):
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
                                    omega=self.activation.omega.item())
            
        # for regular linear layers only if and when we use SIREN init (!)
        if init_siren and self.layer_type in ["last"]:
            self.init_weights_siren(layer_type=self.layer_type, omega=None)

        # edge case WIRE:
        if isinstance(activation, GaborComplexActivation):
            self.linear = nn.Linear(in_features, out_features, dtype=torch.cfloat)

    """ Weight init as in SIREN paper """
    def init_weights_siren(self, layer_type="first", omega=30):    
        with torch.no_grad():
            if layer_type == "first":
                # no division by omega here, as in the paper
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            elif layer_type == "hidden":
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / omega, 
                                             np.sqrt(6 / self.in_features) / omega)
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
    
class GaborComplexActivation(nn.Module):
    """Gabor activation as described in Saragadam WIRE'23.
    """
    def __init__(self, omega=10, sigma=10):
        super().__init__()
        self.register_buffer('omega', torch.tensor(omega))
        self.register_buffer('sigma', torch.tensor(sigma))

    def forward(self, x):
        if not x.is_complex():
            x = x.to(dtype=torch.cfloat)
        omega_lin = self.omega * x
        scale_lin = self.sigma * x
        # Complex Gabor: complex exponential with Gaussian envelope
        # 1:1 as in WIRE paper
        return torch.exp(1j * omega_lin - scale_lin.abs().square())   
    
class ReluFFN(nn.Module):
    """
    Fourier Feature Network (FFN) with ReLU activations.
    """
    def __init__(self, input_dim=2, mapping_size=128, hidden_dim=256,
                 output_dim=3, num_layers=4, sigma=5.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sigma = sigma
        
        # Random Fourier matrix (fixed, NOT learnable)
        B = torch.randn(mapping_size, input_dim) * sigma
        self.register_buffer('B', B)

        # Build MLP
        layers = [ActivatedLinear(2 * mapping_size, hidden_dim, nn.ReLU())]
        for _ in range(num_layers - 2): # 
            layers.append(ActivatedLinear(hidden_dim, hidden_dim, nn.ReLU()))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None))
        self.mlp = nn.Sequential(*layers)

    def fourier_feature_mapping(self, x):
        x_proj = torch.matmul(x, self.B.T)
        return torch.cat([torch.cos(2 * math.pi * x_proj), torch.sin(2 * math.pi * x_proj)], dim=-1)

    def forward(self, x):
        return self.mlp(self.fourier_feature_mapping(x))

    def get_fourier_features_lipschitz_constant(self):
        return 2 * math.pi * torch.linalg.matrix_norm(self.B, ord=2).item()
    
    def get_fourier_features_frobenius_constant(self):
        return 2 * math.pi * torch.linalg.matrix_norm(self.B, ord='fro').item()
    
    def get_fourier_features_vector_norm_constant(self):
        return 2 * math.pi * torch.linalg.matrix_norm(self.B, ord=1).item()
    
    def get_fourier_features_stable_rank(self):
        fro_norm = torch.linalg.matrix_norm(self.B, ord='fro').item()
        spec_norm = torch.linalg.matrix_norm(self.B, ord=2).item()
        return (fro_norm**2) / (spec_norm**2 + 1e-12)
    
    def get_fourier_features_spectral_condition_number(self):
        return torch.linalg.cond(self.B, p=2).item()

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    def get_end_to_end_spectral_bound(self):
        total = self.get_fourier_features_lipschitz_constant()
        for info in self.get_layer_infos():
            total *= info['combined_spectral_norm']
        return total

    def get_detailed_matrix_info(self):
        infos = self.get_layer_infos()
        return {
            'layer_infos': infos,
            'end_to_end_spectral_bound': self.get_end_to_end_spectral_bound(),
            'fourier_features_lipschitz': self.get_fourier_features_lipschitz_constant(),
            'fourier_matrix_spectral_norm': torch.linalg.matrix_norm(self.B, ord=2).item(),
            'fourier_matrix_frobenius_norm': torch.linalg.matrix_norm(self.B, ord='fro').item(),
            'fourier_matrix_stable_rank': self.get_fourier_features_stable_rank(),
            'fourier_matrix_spectral_condition_no': self.get_fourier_features_spectral_condition_number(),
            'fourier_sigma': self.sigma,
            'num_layers': len(infos)
        }

class GaussFFN(nn.Module):
    """
    Gaussian Fourier Feature Network (FFN) with Gaussian activations.
    """
    def __init__(self, input_dim=2, mapping_size=128, hidden_dim=256,
                 output_dim=3, num_layers=4, sigma=5.0, a=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sigma = sigma

        # a could be a list for layer-specific values
        if isinstance(a, (list, tuple)):
            if len(a) != (num_layers -1):
                raise ValueError(f"Length of 'a' list ({len(a)}) must match num_layers -1 ({num_layers -1})")
            self.a_values = a
        else:
            self.a_values = [a] * (num_layers -1)
        
        # Random Fourier matrix (fixed, NOT learnable)
        B = torch.randn(mapping_size, input_dim) * sigma
        self.register_buffer('B', B)

        # Build MLP with layer-specific 'a' values
        layers = [ActivatedLinear(2 * mapping_size, hidden_dim, GaussianActivation(a=self.a_values[0]))]
        for i in range(num_layers - 2):
            layers.append(ActivatedLinear(hidden_dim, hidden_dim, GaussianActivation(a=self.a_values[i+1])))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None))
        self.mlp = nn.Sequential(*layers)

    def fourier_feature_mapping(self, x):
        x_proj = torch.matmul(x, self.B.T)
        return torch.cat([torch.cos(2 * math.pi * x_proj), torch.sin(2 * math.pi * x_proj)], dim=-1)

    def forward(self, x):
        return self.mlp(self.fourier_feature_mapping(x))

    def get_fourier_features_lipschitz_constant(self):
        return 2 * math.pi * torch.linalg.matrix_norm(self.B, ord=2).item()
    
    def get_fourier_features_frobenius_constant(self):
        return 2 * math.pi * torch.linalg.matrix_norm(self.B, ord='fro').item()
    
    def get_fourier_features_stable_rank(self):
        fro_norm = torch.linalg.matrix_norm(self.B, ord='fro').item()
        spec_norm = torch.linalg.matrix_norm(self.B, ord=2).item()
        return (fro_norm**2) / (spec_norm**2 + 1e-12)
    
    def get_fourier_features_spectral_condition_number(self):
        return torch.linalg.cond(self.B, p=2).item()

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    def get_end_to_end_spectral_bound(self):
        total = self.get_fourier_features_lipschitz_constant()
        for info in self.get_layer_infos():
            total *= info['combined_spectral_norm']
        return total

    def get_detailed_matrix_info(self):
        infos = self.get_layer_infos()
        return {
            'layer_infos': infos,
            'end_to_end_spectral_bound': self.get_end_to_end_spectral_bound(),
            'fourier_features_lipschitz': self.get_fourier_features_lipschitz_constant(),
            'fourier_matrix_spectral_norm': torch.linalg.matrix_norm(self.B, ord=2).item(),
            'fourier_matrix_frobenius_norm': torch.linalg.matrix_norm(self.B, ord='fro').item(),
            'fourier_matrix_stable_rank': self.get_fourier_features_stable_rank(),
            'fourier_matrix_spectral_condition_no': self.get_fourier_features_spectral_condition_number(),
            'fourier_sigma': self.sigma,
            'gaussian_a_values': self.a_values,
            'num_layers': len(infos)
        }

class ReluMLP(nn.Module):
    """
    ReLU-activated MLP without FFs.
    """
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # layers = [ActivatedLinear(input_dim, hidden_dim, torch.nn.ReLU())]
        layers = [ActivatedLinear(input_dim, hidden_dim, torch.nn.ReLU())]
        for i in range(num_layers - 2):
            layers.append(ActivatedLinear(hidden_dim, hidden_dim, torch.nn.ReLU()))

        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        # layers.append(ActivatedLinear(hidden_dim, 126, activation=None))
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    def get_end_to_end_spectral_bound(self):
        total = 1.0
        for info in self.get_layer_infos():
            total *= info['combined_spectral_norm']
        return total

    def get_detailed_matrix_info(self):
        infos = self.get_layer_infos()
        return {
            'layer_infos': infos,
            'end_to_end_spectral_bound': self.get_end_to_end_spectral_bound(),
            'num_layers': len(infos)
        }
      
class GaussMLP(nn.Module):
    """
    Gaussian-activated MLP without FFs.
    """
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=4, a=5.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # a could be a list for layer-specific values
        if isinstance(a, (list, tuple)):
            if len(a) != (num_layers-1):
                raise ValueError(f"Length of 'a' list ({len(a)}) must match num_layers -1 ({num_layers-1})")
            self.a_values = a
        else:
            self.a_values = [a] * (num_layers -1)

        # Build MLP with layer-specific 'a' values
        layers = [ActivatedLinear(input_dim, hidden_dim, GaussianActivation(a=self.a_values[0]))]
        for i in range(num_layers - 2):
            layers.append(ActivatedLinear(hidden_dim, hidden_dim, GaussianActivation(a=self.a_values[i+1])))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    def get_end_to_end_spectral_bound(self):
        total = 1.0
        for info in self.get_layer_infos():
            total *= info['combined_spectral_norm']
        return total

    def get_detailed_matrix_info(self):
        infos = self.get_layer_infos()
        return {
            'layer_infos': infos,
            'end_to_end_spectral_bound': self.get_end_to_end_spectral_bound(),
            'gaussian_a_values': self.a_values,
            'num_layers': len(infos)
        }
    
class SirenMLP(nn.Module):
    """
    Sinusoidal-activated MLP as in SIREN, Sitzmann et al. NeurIPS'20.
    """
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=4, omega=30.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # omega could be a list for layer-specific values
        if isinstance(omega, (list, tuple)):
            if len(omega) != (num_layers-1):
                raise ValueError(f"Length of 'omega' list ({len(omega)}) must match num_layers -1 ({num_layers-1})")
            self.omega_values = omega
        else:
            self.omega_values = [omega] * (num_layers -1)

        # Build MLP with layer-specific 'a' values
        layers = [ActivatedLinear(input_dim, hidden_dim, SinusoidalActivation(omega=self.omega_values[0]), layer_type="first", init_siren=True)]
        for i in range(num_layers - 2):
            layers.append(ActivatedLinear(hidden_dim, hidden_dim, SinusoidalActivation(omega=self.omega_values[i+1]), layer_type="hidden", init_siren=True))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        # notice how we use init_siren=True here to trigger the special last-layer init, not the vanilla init for nn.Linear()
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None, layer_type="last", init_siren=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    def get_end_to_end_spectral_bound(self):
        total = 1.0
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

class WireMLP(nn.Module):
    """
    GaborWavelet-activated MLP as in WIRE, Saragadam et al. 
    """
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=4, omega=10.0, sigma=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Reduce hidden features since complex numbers have 2x parameters, according to WIRE paper, to make it fair
        red_hidden_dim = int(hidden_dim / np.sqrt(2))
        self.hidden_dim = red_hidden_dim
        
        # omega could be a list for layer-specific values
        if isinstance(omega, (list, tuple)):
            if len(omega) != (num_layers-1):
                raise ValueError(f"Length of 'omega' list ({len(omega)}) must match num_layers -1 ({num_layers-1})")
            self.omega_values = omega
        else:
            self.omega_values = [omega] * (num_layers -1)
        
        # sigma could be a list for layer-specific values
        if isinstance(sigma, (list, tuple)):
            if len(sigma) != (num_layers-1):
                raise ValueError(f"Length of 'sigma' list ({len(sigma)}) must match num_layers -1 ({num_layers-1})")
            self.sigma_values = sigma
        else:
            self.sigma_values = [sigma] * (num_layers -1)

        # Build MLP with layer-specific 'omega' and 'sigma' values
        layers = [ActivatedLinear(input_dim, red_hidden_dim, GaborComplexActivation(omega=self.omega_values[0],sigma=self.sigma_values[0]), layer_type="first", init_siren=False)]
        for i in range(num_layers - 2):
            # these are complex layers
            layers.append(ActivatedLinear(red_hidden_dim, red_hidden_dim, GaborComplexActivation(omega=self.omega_values[i+1], sigma=self.sigma_values[i+1]), layer_type="hidden", init_siren=False))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        # notice how we use init_siren=True here to trigger the special last-layer init, not the vanilla init for nn.Linear()
        layers.append(ActivatedLinear(red_hidden_dim, output_dim, activation=None, layer_type="wire_last", init_siren=False))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if not x.is_complex():
            x = x.to(dtype=torch.cfloat)
        return self.mlp(x)

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    def get_end_to_end_spectral_bound(self):
        total = 1.0
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

class ReluPosEncoding(nn.Module):
    """
    Positional Encoding Network with ReLU activations.
    """
    def __init__(self, input_dim=2, mapping_size=128, hidden_dim=256,
                 output_dim=3, num_layers=4, sigma=5.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sigma = sigma
        
        # Random Fourier matrix (fixed, NOT learnable)
        B = torch.randn(mapping_size, input_dim) * sigma
        self.register_buffer('B', B)

        # Build MLP
        layers = [ActivatedLinear(2 * mapping_size, hidden_dim, nn.ReLU())]
        for _ in range(num_layers - 2): # 
            layers.append(ActivatedLinear(hidden_dim, hidden_dim, nn.ReLU()))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        layers.append(ActivatedLinear(hidden_dim, output_dim, activation=None))
        self.mlp = nn.Sequential(*layers)

    def fourier_feature_mapping(self, x):
        x_proj = torch.matmul(x, self.B.T)
        return torch.cat([torch.cos(2 * math.pi * x_proj), torch.sin(2 * math.pi * x_proj)], dim=-1)

    def forward(self, x):
        return self.mlp(self.fourier_feature_mapping(x))

    def get_fourier_features_lipschitz_constant(self):
        return 2 * math.pi * torch.linalg.matrix_norm(self.B, ord=2).item()
    
    def get_fourier_features_frobenius_constant(self):
        return 2 * math.pi * torch.linalg.matrix_norm(self.B, ord='fro').item()
    
    def get_fourier_features_vector_norm_constant(self):
        return 2 * math.pi * torch.linalg.matrix_norm(self.B, ord=1).item()
    
    def get_fourier_features_stable_rank(self):
        fro_norm = torch.linalg.matrix_norm(self.B, ord='fro').item()
        spec_norm = torch.linalg.matrix_norm(self.B, ord=2).item()
        return (fro_norm**2) / (spec_norm**2 + 1e-12)
    
    def get_fourier_features_spectral_condition_number(self):
        return torch.linalg.cond(self.B, p=2).item()

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedLinear)]

    def get_end_to_end_spectral_bound(self):
        total = self.get_fourier_features_lipschitz_constant()
        for info in self.get_layer_infos():
            total *= info['combined_spectral_norm']
        return total

    def get_detailed_matrix_info(self):
        infos = self.get_layer_infos()
        return {
            'layer_infos': infos,
            'end_to_end_spectral_bound': self.get_end_to_end_spectral_bound(),
            'fourier_features_lipschitz': self.get_fourier_features_lipschitz_constant(),
            'fourier_matrix_spectral_norm': torch.linalg.matrix_norm(self.B, ord=2).item(),
            'fourier_matrix_frobenius_norm': torch.linalg.matrix_norm(self.B, ord='fro').item(),
            'fourier_matrix_stable_rank': self.get_fourier_features_stable_rank(),
            'fourier_matrix_spectral_condition_no': self.get_fourier_features_spectral_condition_number(),
            'fourier_sigma': self.sigma,
            'num_layers': len(infos)
        }


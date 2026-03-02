import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from deel import torchlip
from k_lipschitz_spectral_norm import spectral_norm
from sll_layers import KLipschitzSDPBasedLipschitzLinearLayer, NormalizedAffineLayer

# import torch.nn.utils.spectral_norm as spectral_norm
# from custom_norms import complex_spectral_norm

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
    elif isinstance(activation,torchlip.GroupSort):
        c= 1.0
    elif isinstance(activation,torchlip.GroupSort2):
        c= 1.0
    elif isinstance(activation,torchlip.HouseHolder):
        c= 1.0
    elif isinstance(activation, SpectralGaussianActivation):
        c = activation.k.item()
    elif isinstance(activation, SpectralSinusoidalActivation):
        c = activation.k.item()
    else:
        raise ValueError(f"Unsupported activation type: {type(activation)}")
    return c

def matrix_info(weight, activation, eps=1e-12):
    """Compute spectral norm, Frobenius norm, stable rank for a weight matrix.
    Expects weight that already has the correct k-scaling applied."""
    act_lip = get_activation_lipschitz(activation)
    
    # Detach from computational graph for analysis
    with torch.no_grad():
        weight_detached = weight.detach()
        spectral_norm = torch.linalg.matrix_norm(weight_detached, ord=2)
        fro_norm = torch.norm(weight_detached, p='fro')
        stable_rank = (fro_norm ** 2) / (spectral_norm ** 2 + eps)
        rms_norm = torch.sqrt(torch.mean(torch.square(torch.norm(weight_detached, p=2, dim=1))))
    
    return {
        'linear_spectral_norm': spectral_norm.item(),  # Should be close to k
        'activation_spectral_norm': act_lip,
        'combined_spectral_norm': spectral_norm.item() * act_lip,
        'frobenius_norm': fro_norm.item(),
        'stable_rank': stable_rank.item(),
        'rms_norm': rms_norm.item(),
    }

class SpectrallyConstrainedLinear(nn.Module):
    """
    Wrap an nn.Linear with a k-Lipschitz variant of torch.nn.utils.spectral_norm.
    """
    def __init__(self, in_features, out_features, bias=True, 
                 n_power_iterations: int = 1, eps: float = 1e-12, dtype=torch.float32, k=None, norm_type="spectral_norm"):
        super().__init__()

        assert k is not None, "Forgot to set k value for layer?"                
        self.register_buffer('k', torch.tensor(float(k)))

        if dtype != torch.cfloat:
            if norm_type == "spectral_norm":
                print("L2 Norm Layer")
                linear = nn.Linear(in_features, out_features, bias=bias)
                self.linear = spectral_norm(linear, n_power_iterations=n_power_iterations, eps=eps, lipschitz_const=float(k))
            elif norm_type=="bjoerck":
                print("Deel Torchlip Bjoerck")
                self.linear = torchlip.SpectralLinear(in_features,out_features, bias=bias, k_coef_lip=float(k))  
            elif norm_type=="sll":
                # as SLL is residual, only hidden layers can be used like this
                if in_features == out_features:
                    self.linear = KLipschitzSDPBasedLipschitzLinearLayer(in_features, out_features, k_coef_lip=float(k))  
                elif in_features > out_features:
                    self.linear = torchlip.SpectralLinear(in_features=in_features,out_features=out_features,k_coef_lip=float(k))
                elif in_features < out_features:
                    self.linear = torchlip.SpectralLinear(in_features=in_features,out_features=out_features,k_coef_lip=float(k))
                else:
                    raise NotImplementedError("Complex dtype not yet supported with k-Lipschitz spectral norm.")
        else:
            raise NotImplementedError("Complex dtype not yet supported with k-Lipschitz spectral norm.")

    def get_effective_weight(self):
        """Get the effective weight used in forward pass (already k-scaled)"""
        with torch.no_grad():
            # With custom k-Lipschitz spectral norm, the weight is already k-scaled
            return self.linear.weight
        
    def forward(self, x):
        return self.linear(x)

class ActivatedSpectralLinear(nn.Module):
    """Linear layer with activation and spectral norm tracking."""
    def __init__(self, in_features, out_features, activation=None, layer_type="hidden", init_siren=False, k=None, dtype=torch.float32, norm_type="norm"):
        super().__init__()

        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        self.layer_type = layer_type
        self.register_buffer('k', torch.tensor(float(k)))

        assert self.k is not None, "Forgot to set k value for layer?"

        # in case of wire last layer, we need complex dtype
        self.linear = SpectrallyConstrainedLinear(in_features, out_features, k=k, dtype=dtype, norm_type=norm_type)

        # edge case SIREN:
        # we do this for all sinusoidal activations
        if isinstance(activation, SpectralSinusoidalActivation):
            self.init_weights_siren(layer_type=self.layer_type,
                                    omega=self.activation.omega.item())

        # for last layer if we use SIREN init (!)
        if init_siren and self.layer_type in ["last"]:
            self.init_weights_siren(layer_type=self.layer_type, omega=None)

    """ Weight init as in SIREN paper """
    def init_weights_siren(self, layer_type="first", omega=30):    
        with torch.no_grad():
            if layer_type == "first":
                # no division by omega here, as in the paper
                self.linear.linear.weight.uniform_(-1 / self.in_features, 
                                                1 / self.in_features)      
            elif layer_type == "hidden":
                self.linear.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / omega, 
                                                np.sqrt(6 / self.in_features) / omega)
            elif layer_type == "last":
                # in SIREN the dim is hidden_features, but the in_features are hidden_features (!)
                # notice how we don't divide by omega here as well
                self.linear.linear.weight.uniform_(-np.sqrt(6 / self.in_features), 
                                                np.sqrt(6 / self.in_features))

    def forward(self, x):
        if self.activation is None:
            return self.linear(x)
        else:           
            return self.activation(self.linear(x))

    def get_info(self):
        with torch.no_grad():
            effective_weight = self.linear.get_effective_weight()
        return matrix_info(effective_weight.clone(), self.activation)
     
class SpectralSinusoidalActivation(nn.Module):
    """Sinusoidal activation function as described in SIREN, Sitzmann et al. NeurIPS'20.
    
    Implements the sinusoidal activation: sin(omega * x).
    Here however instead of setting omega, we set the Lipschitz constant and compute omega from it.
    The Lipschitz constant of sin(omega * x) is omega. This is trival since setting k is equivalent to setting omega.
    """
    def __init__(self, k=1.0):
        super().__init__()
        # compute omega from Lipschitz constant k, simple in this case
        omega = float(k)
        self.register_buffer('omega', torch.tensor(omega))
        self.register_buffer('k', torch.tensor(k))

    def forward(self, x):
        return torch.sin(self.omega * x)
    
class SpectralGaussianActivation(nn.Module):
    """Gaussian activation function as described in Table 1, Beyond Periodicity: Towards a 
    Unifying Framework for Activations in Coordinate-MLPs, Ramasingh et al. ECCV'22.
    
    Implements the Gaussian activation: exp(-x²/(2a²)).

    Here however instead of setting a, we set the Lipschitz constant and compute a from it.
    The Lipschitz constant of exp(-x²/(2a²)) is 1 / (activation.a*math.sqrt(math.e))
    """
    def __init__(self, k=1.0):
        super().__init__()
        a = 1/(k* math.sqrt(math.e))
        self.register_buffer('a', torch.tensor(a))
        self.register_buffer('k', torch.tensor(k))
        self.register_buffer('omega',torch.tensor(k))

    def forward(self, x):
        return torch.exp(-0.5 * x**2 / (self.a**2))

class ReluFFN(nn.Module):
    """
    Fourier Feature Network (FFN) with ReLU activations.
    """
    def __init__(self, input_dim=2, mapping_size=32, hidden_dim=256,
                 output_dim=3, num_layers=4, k_layers=1.0, k_ff=1.0, norm_type="spectral_norm"):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_type = norm_type

        # k could be a list for layer-specific values
        if isinstance(k_layers, (list, tuple)):
            if len(k_layers) != (num_layers):
                raise ValueError(f"Length of 'k_layers' list ({len(k_layers)}) must match num_layers ({num_layers})")
            self.k_layers_values = k_layers
        else:
            self.k_layers_values = [k_layers] * num_layers



        self.sigma = k_ff / (2*torch.pi*math.sqrt(mapping_size))
        
        # Random Fourier matrix (fixed, NOT learnable)
        B = torch.randn(mapping_size, input_dim) * self.sigma
        self.register_buffer('B', B)

        print("lipschitz constant to be set:", k_ff)
        print("mapping size:", mapping_size)
        print("calucalted sigma:",self.sigma)
        print("measured lipschitz", self.get_fourier_features_lipschitz_constant())

        # Build MLP
        layers = [ActivatedSpectralLinear(2 * mapping_size, hidden_dim, activation=nn.ReLU(), k=self.k_layers_values[0], norm_type=self.norm_type)]
        for i in range(num_layers - 2): # 
            layers.append(ActivatedSpectralLinear(hidden_dim, hidden_dim, activation=nn.ReLU(),k=self.k_layers_values[i+1],norm_type=self.norm_type))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        layers.append(ActivatedSpectralLinear(hidden_dim, output_dim, activation=None, k=self.k_layers_values[-1],norm_type=self.norm_type))
        self.mlp = nn.Sequential(*layers)

    def fourier_feature_mapping(self, x):
        x_proj = torch.matmul(x, self.B.T)
        return torch.cat([torch.cos(2 * math.pi * x_proj), torch.sin(2 * math.pi * x_proj)], dim=-1)

    def forward(self, x):
        return self.mlp(self.fourier_feature_mapping(x))

    def get_fourier_features_lipschitz_constant(self):
        return 2 * math.pi * torch.linalg.matrix_norm(self.B, ord=2).item()

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedSpectralLinear)]

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
            'fourier_matrix_rms_norm': 2.0 * np.pi * torch.sqrt(torch.mean(torch.square(torch.norm(self.B, p=2, dim=1)))).item()
        }

class GaussFFN(nn.Module):
    """
    Gaussian Fourier Feature Network (FFN) with Gaussian activations.
    """
    def __init__(self, input_dim=2, mapping_size=128, hidden_dim=256,
                 output_dim=3, num_layers=4, k_layers = 1.0, k_act=1.0, k_ff=1.0, norm_type="spectral_norm"):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_type = norm_type

        # could be a list for layer-specific values
        if isinstance(k_layers, (list, tuple)):
            if len(k_layers) != (num_layers):
                raise ValueError(f"Length of 'k_layers' list ({len(k_layers)}) must match num_layers ({num_layers})")
            self.k_layers_values = k_layers
        else:
            self.k_layers_values = [k_layers] * num_layers

        # k_act could be a list for layer-specific values
        if isinstance(k_act, (list, tuple)):
            if len(k_act) != (num_layers -1):
                raise ValueError(f"Length of 'a' list ({len(k_act)}) must match num_layers -1 ({num_layers -1})")
            self.k_act_values = k_act
        else:
            self.k_act_values = [k_act] * (num_layers -1)
        
        # we need to infer the sigma from k_ff and the mapping size
        self.sigma = k_ff / (2*torch.pi*math.sqrt(mapping_size))
        
        # Random Fourier matrix (fixed, NOT learnable)
        B = torch.randn(mapping_size, input_dim) * self.sigma
        self.register_buffer('B', B)

        # Build MLP with layer-specific 'a' values
        layers = [ActivatedSpectralLinear(2 * mapping_size, hidden_dim, SpectralGaussianActivation(k=self.k_act_values[0]), k=self.k_layers_values[0], norm_type=self.norm_type)]
        for i in range(num_layers - 2):
            layers.append(ActivatedSpectralLinear(hidden_dim, hidden_dim, SpectralGaussianActivation(k=self.k_act_values[i+1]), k=self.k_layers_values[i+1], norm_type=self.norm_type))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        layers.append(ActivatedSpectralLinear(hidden_dim, output_dim, activation=None, k=self.k_layers_values[-1], norm_type=self.norm_type))
        self.mlp = nn.Sequential(*layers)

    def fourier_feature_mapping(self, x):
        x_proj = torch.matmul(x, self.B.T)
        return torch.cat([torch.cos(2 * math.pi * x_proj), torch.sin(2 * math.pi * x_proj)], dim=-1)

    def forward(self, x):
        return self.mlp(self.fourier_feature_mapping(x))

    def get_fourier_features_lipschitz_constant(self):
        return 2 * math.pi * torch.linalg.matrix_norm(self.B, ord=2).item()

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedSpectralLinear)]

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
            'fourier_matrix_rms_norm': 2.0 * np.pi * torch.sqrt(torch.mean(torch.square(torch.norm(self.B, p=2, dim=1)))).item(),
            'fourier_sigma': self.sigma,
            'gaussian_k_values': self.k_act_values,
            'num_layers': len(infos)
        }
    
class ReluMLP(nn.Module):
    """
    ReLU-activated MLP without FFs.
    """
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=4, k_layers=1.0, norm_type="spectral_norm"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_type= norm_type

        # a could be a list for layer-specific values
        if isinstance(k_layers, (list, tuple)):
            if len(k_layers) != (num_layers):
                raise ValueError(f"Length of 'k_layers' list ({len(k_layers)}) must match num_layers ({num_layers})")
            self.k_layers_values = k_layers
        else:
            self.k_layers_values = [k_layers] * num_layers
        
        layers = [ActivatedSpectralLinear(input_dim, hidden_dim, torch.nn.ReLU(), k=self.k_layers_values[0], norm_type=self.norm_type)]
        for i in range(num_layers - 2):
            layers.append(ActivatedSpectralLinear(hidden_dim, hidden_dim, torch.nn.ReLU(), k=self.k_layers_values[i+1], norm_type=self.norm_type))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        layers.append(ActivatedSpectralLinear(hidden_dim, output_dim, activation=None, k=self.k_layers_values[-1], norm_type=self.norm_type))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedSpectralLinear)]

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
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=4, k_layers=1.0, k_act=1.0, norm_type="spectral_norm"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_type= norm_type
        
        # could be a list for layer-specific values
        if isinstance(k_layers, (list, tuple)):
            if len(k_layers) != (num_layers):
                raise ValueError(f"Length of 'k_layers' list ({len(k_layers)}) must match num_layers ({num_layers})")
            self.k_layers_values = k_layers
        else:
            self.k_layers_values = [k_layers] * num_layers

        # a could be a list for layer-specific values
        if isinstance(k_act, (list, tuple)):
            if len(k_act) != (num_layers -1):
                raise ValueError(f"Length of 'a' list ({len(k_act)}) must match num_layers -1 ({num_layers -1})")
            self.k_act_values = k_act
        else:
            self.k_act_values = [k_act] * (num_layers -1)
        
        # Build MLP with layer-specific 'a' values
        layers = [ActivatedSpectralLinear(input_dim, hidden_dim, SpectralGaussianActivation(k=self.k_act_values[0]),k=self.k_layers_values[0], norm_type=self.norm_type)]
        for i in range(num_layers - 2):
            layers.append(ActivatedSpectralLinear(hidden_dim, hidden_dim, SpectralGaussianActivation(k=self.k_act_values[i+1]), k=self.k_layers_values[i+1], norm_type=self.norm_type))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        layers.append(ActivatedSpectralLinear(hidden_dim, output_dim, activation=None, k=self.k_layers_values[-1], norm_type=self.norm_type))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedSpectralLinear)]

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
            'gaussian_k_values': self.k_act_values,
            'num_layers': len(infos)
        }
    
class SirenMLP(nn.Module):
    """
    Sinusoidal-activated MLP as in SIREN, Sitzmann et al. NeurIPS'20.
    """
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=4, k_layers=1.0, k_act=1.0, norm_type="spectral_norm"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_type= norm_type
        
        # could be a list for layer-specific values
        if isinstance(k_layers, (list, tuple)):
            if len(k_layers) != (num_layers):
                raise ValueError(f"Length of 'k_layers' list ({len(k_layers)}) must match num_layers ({num_layers})")
            self.k_layers_values = k_layers
        else:
            self.k_layers_values = [k_layers] * num_layers

        # a could be a list for layer-specific values
        if isinstance(k_act, (list, tuple)):
            if len(k_act) != (num_layers -1):
                raise ValueError(f"Length of 'a' list ({len(k_act)}) must match num_layers -1 ({num_layers -1})")
            self.k_act_values = k_act
        else:
            self.k_act_values = [k_act] * (num_layers -1)

        print(self.k_act_values)
        print(self.k_layers_values)

        # Build MLP with layer-specific 'a' values
        layers = [ActivatedSpectralLinear(input_dim, hidden_dim, SpectralSinusoidalActivation(k=self.k_act_values[0]), layer_type="first", init_siren=True, k=self.k_layers_values[0], norm_type=self.norm_type)]
        for i in range(num_layers - 2):
            layers.append(ActivatedSpectralLinear(hidden_dim, hidden_dim, SpectralSinusoidalActivation(k=self.k_act_values[i+1]), layer_type="hidden", init_siren=True, k=self.k_layers_values[i+1], norm_type=self.norm_type))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        # notice how we use init_siren=True here to trigger the special last-layer init, not the vanilla init for nn.Linear()
        layers.append(ActivatedSpectralLinear(hidden_dim, output_dim, activation=None, layer_type="last", init_siren=True, k=self.k_layers_values[-1], norm_type=self.norm_type))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedSpectralLinear)]

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
            'omega_k_values': self.k_act_values,
            'num_layers': len(infos)
        }

class GroupSortFFN(nn.Module):
    """
    Fourier Feature Network (FFN) with GroupSort activations.
    """
    def __init__(self, input_dim=2, mapping_size=32, hidden_dim=256,
                 output_dim=3, num_layers=4, k_layers=1.0, k_ff=1.0, group_size=2, norm_type="spectral_norm"):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_type= norm_type

        # k could be a list for layer-specific values
        if isinstance(k_layers, (list, tuple)):
            if len(k_layers) != (num_layers):
                raise ValueError(f"Length of 'k_layers' list ({len(k_layers)}) must match num_layers ({num_layers})")
            self.k_layers_values = k_layers
        else:
            self.k_layers_values = [k_layers] * num_layers

        print("k_ff:", k_ff)
        print("mapping size:", mapping_size)

        self.sigma = k_ff / (2*torch.pi*math.sqrt(mapping_size))
        
        # Random Fourier matrix (fixed, NOT learnable)
        B = torch.randn(mapping_size, input_dim) * self.sigma
        self.register_buffer('B', B)

        print(self.sigma)
        print(self.get_fourier_features_lipschitz_constant())

        # Build MLP
        layers = [ActivatedSpectralLinear(2 * mapping_size, hidden_dim, activation=torchlip.GroupSort(group_size=group_size), k=self.k_layers_values[0], norm_type=self.norm_type)]
        for i in range(num_layers - 2): # 
            layers.append(ActivatedSpectralLinear(hidden_dim, hidden_dim, activation=torchlip.GroupSort(group_size=group_size),k=self.k_layers_values[i+1], norm_type=self.norm_type))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        layers.append(ActivatedSpectralLinear(hidden_dim, output_dim, activation=None, k=self.k_layers_values[-1], norm_type=self.norm_type))
        self.mlp = nn.Sequential(*layers)

    def fourier_feature_mapping(self, x):
        x_proj = torch.matmul(x, self.B.T)
        return torch.cat([torch.cos(2 * math.pi * x_proj), torch.sin(2 * math.pi * x_proj)], dim=-1)

    def forward(self, x):
        return self.mlp(self.fourier_feature_mapping(x))

    def get_fourier_features_lipschitz_constant(self):
        return 2 * math.pi * torch.linalg.matrix_norm(self.B, ord=2).item()

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedSpectralLinear)]

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
            'fourier_matrix_rms_norm': 2.0 * np.pi * torch.sqrt(torch.mean(torch.square(torch.norm(self.B, p=2, dim=1)))).item(),
            'num_layers': len(infos)
        }
    
class GroupSortMLP(nn.Module):
    """
    ReLU-activated MLP without FFs.
    """
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=4, k_layers=1.0, norm_type="spectral_norm", group_size=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_type= norm_type

        print(group_size)

        # a could be a list for layer-specific values
        if isinstance(k_layers, (list, tuple)):
            if len(k_layers) != (num_layers):
                raise ValueError(f"Length of 'k_layers' list ({len(k_layers)}) must match num_layers ({num_layers})")
            self.k_layers_values = k_layers
        else:
            self.k_layers_values = [k_layers] * num_layers
        
        layers = [ActivatedSpectralLinear(input_dim, hidden_dim, torchlip.GroupSort(group_size=group_size), k=self.k_layers_values[0],norm_type=self.norm_type)]
        for i in range(num_layers - 2):
            layers.append(ActivatedSpectralLinear(hidden_dim, hidden_dim, torchlip.GroupSort(group_size=group_size), k=self.k_layers_values[i+1], norm_type=self.norm_type))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        layers.append(ActivatedSpectralLinear(hidden_dim, output_dim, activation=None, k=self.k_layers_values[-1], norm_type=self.norm_type))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedSpectralLinear)]

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
    
class ReluPosEncoding(nn.Module):
    """
    Fourier Feature Network (FFN) with ReLU activations.
    """
    def __init__(self, input_dim=2, mapping_size=32, hidden_dim=256,
                 output_dim=3, num_layers=4, k_layers=1.0, k_ff=1.0, norm_type="spectral_norm"):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_type = norm_type

        # k could be a list for layer-specific values
        if isinstance(k_layers, (list, tuple)):
            if len(k_layers) != (num_layers):
                raise ValueError(f"Length of 'k_layers' list ({len(k_layers)}) must match num_layers ({num_layers})")
            self.k_layers_values = k_layers
        else:
            self.k_layers_values = [k_layers] * num_layers



        self.sigma = k_ff / (2*torch.pi*math.sqrt(mapping_size))
        
        # Random Fourier matrix (fixed, NOT learnable)
        B = torch.randn(mapping_size, input_dim) * self.sigma
        self.register_buffer('B', B)

        print("lipschitz constant to be set:", k_ff)
        print("mapping size:", mapping_size)
        print("calucalted sigma:",self.sigma)
        print("measured lipschitz", self.get_fourier_features_lipschitz_constant())

        # Build MLP
        layers = [ActivatedSpectralLinear(2 * mapping_size, hidden_dim, activation=nn.ReLU(), k=self.k_layers_values[0], norm_type=self.norm_type)]
        for i in range(num_layers - 2): # 
            layers.append(ActivatedSpectralLinear(hidden_dim, hidden_dim, activation=nn.ReLU(),k=self.k_layers_values[i+1],norm_type=self.norm_type))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        layers.append(ActivatedSpectralLinear(hidden_dim, output_dim, activation=None, k=self.k_layers_values[-1],norm_type=self.norm_type))
        self.mlp = nn.Sequential(*layers)

    def fourier_feature_mapping(self, x):
        x_proj = torch.matmul(x, self.B.T)
        return torch.cat([torch.cos(2 * math.pi * x_proj), torch.sin(2 * math.pi * x_proj)], dim=-1)

    def forward(self, x):
        return self.mlp(self.fourier_feature_mapping(x))

    def get_fourier_features_lipschitz_constant(self):
        return 2 * math.pi * torch.linalg.matrix_norm(self.B, ord=2).item()

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedSpectralLinear)]

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
            'fourier_matrix_rms_norm': 2.0 * np.pi * torch.sqrt(torch.mean(torch.square(torch.norm(self.B, p=2, dim=1)))).item()
        }

class PositionalEncodingReluMlp(nn.Module):
    """
    Relu Network with Positional Encoding
    """
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=4, k_layers=1.0, k_ff=1.0, norm_type="spectral_norm"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_type = norm_type

        # k could be a list for layer-specific values
        if isinstance(k_layers, (list, tuple)):
            if len(k_layers) != (num_layers):
                raise ValueError(f"Length of 'k_layers' list ({len(k_layers)}) must match num_layers ({num_layers})")
            self.k_layers_values = k_layers
        else:
            self.k_layers_values = [k_layers] * num_layers

        # Calculate L from the desired Lipschitz constant k_ff
        self.pos_l = self.solve_for_L(k_ff)
        
        # Calculate the actual mapping size: input_dim * 2 * L
        # For each input dimension, we get 2*L features (L sin + L cos)
        self.mapping_size = self.input_dim * 2 * int(self.pos_l.item())
        
        # Calculate the Lipschitz constant K from the formula
        self.lipschitz_pos_encoding = (torch.pi * torch.sqrt((4**torch.tensor(self.pos_l) - 1) / 3)).item()

        print("desired lipschitz constant:", k_ff)
        print("calculated L (number of frequency levels):", self.pos_l)
        print("mapping size (total positional encoding features):", self.mapping_size)
        print("calculated lipschitz constant K from formula:", self.lipschitz_pos_encoding)

        # Build MLP - first layer takes mapping_size input
        layers = [ActivatedSpectralLinear(self.mapping_size, hidden_dim, activation=nn.ReLU(), k=self.k_layers_values[0], norm_type=self.norm_type)]
        for i in range(num_layers - 2):
            layers.append(ActivatedSpectralLinear(hidden_dim, hidden_dim, activation=nn.ReLU(), k=self.k_layers_values[i+1], norm_type=self.norm_type))
        # Final layer without activation
        layers.append(ActivatedSpectralLinear(hidden_dim, output_dim, activation=None, k=self.k_layers_values[-1], norm_type=self.norm_type))
        self.mlp = nn.Sequential(*layers)

    def solve_for_L(self, K):
        """
        Calculates L using the formula L = ln(3 * (K/pi)^2 + 1) / ln(4).

        Args:
            K (torch.Tensor or float): The value(s) of K.
                                    Can be a single value or a tensor.

        Returns:
            torch.Tensor: The calculated value(s) of L.
        """
        # Ensure K is a PyTorch tensor for consistent operations
        if not isinstance(K, torch.Tensor):
            K = torch.tensor(K, dtype=torch.float32)

        # Use torch.pi for the value of pi
        pi_tensor = torch.pi
        
        # Calculate the term inside the natural logarithm: 3 * (K/pi)^2 + 1
        # torch.pow can handle both single values and tensors
        log_arg = 3 * torch.pow(K / pi_tensor, 2) + 1
        
        # Calculate L using the natural logarithm (torch.log)
        pos_l = torch.log(log_arg) / torch.log(torch.tensor(4.0))
        
        return torch.ceil(pos_l)

    def positional_encoding(self, p):
        """
        γ(p) = [sin(2^0πp), cos(2^0πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]
        """
        p_expanded = p.unsqueeze(-1)  # Shape: (..., input_dim, 1)
        
        frequencies = 2 ** torch.arange(int(self.pos_l.item()), device=p.device, dtype=p.dtype)
        
        freq_p = frequencies * torch.pi * p_expanded
        
        sin_p = torch.sin(freq_p)
        cos_p = torch.cos(freq_p)
        
        gamma_p = torch.stack([sin_p, cos_p], dim=-1).flatten(start_dim=-2)
        
        return gamma_p.flatten(start_dim=-2)

    def forward(self, x):
        return self.mlp(self.positional_encoding(x))

    def get_end_to_end_spectral_bound(self):
        total = self.lipschitz_pos_encoding
        for info in self.get_layer_infos():
            total *= info['combined_spectral_norm']
        return total

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedSpectralLinear)]

    def get_detailed_matrix_info(self):
        infos = self.get_layer_infos()
        return {
            'layer_infos': infos,
            'end_to_end_spectral_bound': self.get_end_to_end_spectral_bound(),
            'positional_encoding_lipschitz': self.lipschitz_pos_encoding
        }
    
class HouseholderMLP(nn.Module):
    """
    ReLU-activated MLP without FFs.
    """
    def __init__(self, input_dim=2, hidden_dim=256, output_dim=3, num_layers=4, k_layers=1.0, norm_type="spectral_norm"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.norm_type= norm_type
        # a could be a list for layer-specific values
        if isinstance(k_layers, (list, tuple)):
            if len(k_layers) != (num_layers):
                raise ValueError(f"Length of 'k_layers' list ({len(k_layers)}) must match num_layers ({num_layers})")
            self.k_layers_values = k_layers
        else:
            self.k_layers_values = [k_layers] * num_layers
        
        layers = [ActivatedSpectralLinear(input_dim, hidden_dim, torchlip.HouseHolder(channels=2), k=self.k_layers_values[0],norm_type=self.norm_type)]
        for i in range(num_layers - 2):
            layers.append(ActivatedSpectralLinear(hidden_dim, hidden_dim, torchlip.HouseHolder(channels=2), k=self.k_layers_values[i+1], norm_type=self.norm_type))
        # DO NOT CHANGE: we use ActivatedLinear to get the spectral norm info, even without activation
        layers.append(ActivatedSpectralLinear(hidden_dim, output_dim, activation=None, k=self.k_layers_values[-1], norm_type=self.norm_type))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

    def get_layer_infos(self):
        return [m.get_info() for m in self.mlp if isinstance(m, ActivatedSpectralLinear)]

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
"""
S4 (Structured State Space) Layer Implementation for ABR Signal Processing

Based on the high-quality implementation from SSSD-ECG project.
Implements efficient state-space models for long sequence modeling.

Reference: "Efficiently Modeling Long Sequences with Structured State Spaces"
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from scipy import special as ss
from einops import rearrange, repeat
import opt_einsum as oe

contract = oe.contract

def activation_fn(name: str):
    """Get activation function by name."""
    if name == 'gelu':
        return nn.GELU()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'silu' or name == 'swish':
        return nn.SiLU()
    elif name == 'tanh':
        return nn.Tanh()
    else:
        return nn.GELU()  # Default


def get_initializer(name: str, activation: Optional[str] = None):
    """Get weight initializer function."""
    if activation in [None, 'id', 'identity', 'linear']:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu'
    else:
        raise NotImplementedError(f"Activation {activation} not supported")

    if name == 'uniform':
        return nn.init.kaiming_uniform_
    elif name == 'normal':
        return nn.init.kaiming_normal_
    elif name == 'xavier':
        return nn.init.xavier_normal_
    elif name == 'zero':
        return lambda x: nn.init.constant_(x, 0)
    elif name == 'one':
        return lambda x: nn.init.constant_(x, 1)
    else:
        raise NotImplementedError(f"Initializer {name} not supported")


def activation_fn(activation: Optional[str] = None):
    """Get activation function."""
    if activation in [None, 'id', 'identity', 'linear']:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError(f"Activation {activation} not implemented")


class TransposedLinear(nn.Module):
    """Linear module on the second-to-last dimension."""
    
    def __init__(self, d_input: int, d_output: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_output, d_input))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(d_output, 1))
            bound = 1 / math.sqrt(d_input)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = 0.0
    
    def forward(self, x):
        return contract('... u l, v u -> ... v l', x, self.weight) + self.bias


def LinearActivation(
    d_input: int, 
    d_output: int, 
    bias: bool = True,
    zero_bias_init: bool = False,
    transposed: bool = False,
    initializer: Optional[str] = None,
    activation: Optional[str] = None,
    activate: bool = False,
    weight_norm: bool = False,
    **kwargs,
):
    """Create linear layer with flexible activation and initialization."""
    linear_cls = TransposedLinear if transposed else nn.Linear
    if activation == 'glu': 
        d_output *= 2
    
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)
    
    # Initialize weight
    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)
    
    # Initialize bias
    if bias and zero_bias_init:
        nn.init.zeros_(linear.bias)
    
    # Weight norm
    if weight_norm:
        linear = nn.utils.weight_norm(linear)
    
    if activate and activation is not None:
        activation_layer = activation_fn(activation)
        linear = nn.Sequential(linear, activation_layer)
    
    return linear


def transition(measure: str, N: int, **measure_args):
    """Generate A, B transition matrices for different measures."""
    if measure == 'legs':  # Legendre (scaled)
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy()
    elif measure == 'legt':  # Legendre (translated)
        Q = np.arange(N, dtype=np.float64)
        R = (2*Q + 1) ** .5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
        B = R[:, None]
        A = -A
    elif measure == 'lagt':  # Laguerre (translated)
        b = measure_args.get('beta', 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = b * np.ones((N, 1))
    else:
        raise NotImplementedError(f"Measure {measure} not implemented")
    
    return A, B


def rank_correction(measure: str, N: int, rank: int = 1, dtype=torch.float):
    """Return low-rank matrix L such that A + L is normal."""
    if measure == 'legs':
        assert rank >= 1
        P = torch.sqrt(.5 + torch.arange(N, dtype=dtype)).unsqueeze(0)
    elif measure == 'legt':
        assert rank >= 2
        P = torch.sqrt(1 + 2*torch.arange(N, dtype=dtype))
        P0 = P.clone()
        P0[0::2] = 0.
        P1 = P.clone()
        P1[1::2] = 0.
        P = torch.stack([P0, P1], dim=0)
    elif measure == 'lagt':
        assert rank >= 1
        P = .5**.5 * torch.ones(1, N, dtype=dtype)
    else:
        raise NotImplementedError(f"Measure {measure} not supported")
    
    d = P.size(0)
    if rank > d:
        P = torch.cat([P, torch.zeros(rank-d, N, dtype=dtype)], dim=0)
    
    return P


def nplr(measure: str, N: int, rank: int = 1, dtype=torch.float):
    """Return w, p, q, V, B for NPLR representation."""
    A, B = transition(measure, N)
    A = torch.as_tensor(A, dtype=dtype)
    B = torch.as_tensor(B, dtype=dtype)[:, 0]
    
    P = rank_correction(measure, N, rank=rank, dtype=dtype)
    AP = A + torch.sum(P.unsqueeze(-2)*P.unsqueeze(-1), dim=-3)
    w, V = torch.linalg.eig(AP)
    
    # Only keep one of the conjugate pairs
    w = w[..., 0::2].contiguous()
    V = V[..., 0::2].contiguous()
    
    V_inv = V.conj().transpose(-1, -2)
    B = contract('ij, j -> i', V_inv, B.to(V))
    P = contract('ij, ...j -> ...i', V_inv, P.to(V))
    
    return w, P, B, V


def power(L: int, A: torch.Tensor, v: Optional[torch.Tensor] = None):
    """Compute A^L and the scan sum_i A^i v_i."""
    I = torch.eye(A.shape[-1]).to(A)
    
    powers = [A]
    l = 1
    while True:
        if L % 2 == 1: 
            I = powers[-1] @ I
        L //= 2
        if L == 0: 
            break
        l *= 2
        powers.append(powers[-1] @ powers[-1])
    
    if v is None: 
        return I
    
    # Handle reduction for power of 2
    k = v.size(-1) - l
    v_ = powers.pop() @ v[..., l:]
    v = v[..., :l]
    v[..., :k] = v[..., :k] + v_
    
    while v.size(-1) > 1:
        v = rearrange(v, '... (z l) -> ... z l', z=2)
        v = v[..., 0, :] + powers.pop() @ v[..., 1, :]
    
    return I, v.squeeze(-1)


def cauchy_slow(v: torch.Tensor, z: torch.Tensor, w: torch.Tensor):
    """Slow fallback Cauchy kernel computation."""
    cauchy_matrix = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))
    return torch.sum(cauchy_matrix, dim=-2)


_c2r = torch.view_as_real
_r2c = torch.view_as_complex
_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()


class SSKernelNPLR(nn.Module):
    """
    Stores and computes the SSKernel function K_L(A^dt, B^dt, C)
    corresponding to a discretized state space with Normal + Low Rank (NPLR) A matrix.
    """
    
    def __init__(
        self,
        L: int,
        w: torch.Tensor,
        P: torch.Tensor, 
        B: torch.Tensor,
        C: torch.Tensor,
        log_dt: torch.Tensor,
        hurwitz: bool = False,
        trainable: Optional[dict] = None,
        lr: Optional[float] = None,
        tie_state: bool = False,
        length_correction: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        self.hurwitz = hurwitz
        self.tie_state = tie_state
        self.verbose = verbose
        
        # Dimensions
        self.rank = P.shape[-2]
        assert w.size(-1) == P.size(-1) == B.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = w.size(-1)
        
        # Broadcast to correct shapes
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N)))
        H = 1 if self.tie_state else self.H
        B = repeat(B, 'n -> 1 h n', h=H)
        P = repeat(P, 'r n -> r h n', h=H)
        w = repeat(w, 'n -> h n', h=H)
        
        # Cache FFT nodes
        self.L = L
        if self.L is not None:
            self._omega(self.L, dtype=C.dtype, device=C.device, cache=True)
        
        # Register parameters
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))
        
        train = False
        if trainable is None: 
            trainable = {}
        if trainable == False: 
            trainable = {}
        if trainable == True: 
            trainable, train = {}, True
        
        self.register("log_dt", log_dt, trainable.get('dt', train), lr, 0.0)
        self.register("B", _c2r(B), trainable.get('B', train), lr, 0.0)
        self.register("P", _c2r(P), trainable.get('P', train), lr, 0.0)
        
        if self.hurwitz:
            log_w_real = torch.log(-w.real + 1e-3)
            w_imag = w.imag
            self.register("log_w_real", log_w_real, trainable.get('A', 0), lr, 0.0)
            self.register("w_imag", w_imag, trainable.get('A', train), lr, 0.0)
            self.Q = None
        else:
            self.register("w", _c2r(w), trainable.get('A', train), lr, 0.0)
            Q = _resolve_conj(P.clone())
            self.register("Q", _c2r(Q), trainable.get('P', train), lr, 0.0)
        
        if length_correction:
            self._setup_C()
    
    def _omega(self, L: int, dtype, device, cache: bool = True):
        """Calculate FFT nodes and apply bilinear transform."""
        omega = torch.tensor(
            np.exp(-2j * np.pi / L), dtype=dtype, device=device
        )
        omega = omega ** torch.arange(0, L // 2 + 1, device=device)
        z = 2 * (1 - omega) / (1 + omega)
        
        if cache:
            self.register_buffer("omega", _c2r(omega))
            self.register_buffer("z", _c2r(z))
        
        return omega, z
    
    def _w(self):
        """Get the internal w (diagonal) parameter."""
        if self.hurwitz:
            w_real = -torch.exp(self.log_w_real)
            w_imag = self.w_imag
            w = w_real + 1j * w_imag
        else:
            w = _r2c(self.w)
        return w
    
    @torch.no_grad()
    def _setup_state(self):
        """Construct dA and dB for discretized state equation."""
        self._setup_linear()
        C = _r2c(self.C)
        
        state = torch.eye(2*self.N, dtype=C.dtype, device=C.device).unsqueeze(-2)
        dA = self._step_state_linear(state=state)
        dA = rearrange(dA, "n h m -> h m n")
        self.dA = dA
        
        u = C.new_ones(self.H)
        dB = self._step_state_linear(u=u)
        dB = _conj(dB)
        self.dB = rearrange(dB, '1 h n -> h n')
    
    @torch.no_grad()
    def _setup_C(self, double_length: bool = False):
        """Construct C~ from C."""
        C = _r2c(self.C)
        self._setup_state()
        dA_L = power(self.L, self.dA)
        
        C_ = _conj(C)
        prod = contract("h m n, c h n -> c h m", dA_L.transpose(-1, -2), C_)
        if double_length: 
            prod = -prod
        C_ = C_ - prod
        C_ = C_[..., :self.N]
        self.C.copy_(_c2r(C_))
        
        if double_length:
            self.L *= 2
            self._omega(self.L, dtype=C.dtype, device=C.device, cache=True)
    
    def _setup_linear(self):
        """Create parameters for fast linear stepping."""
        w = self._w()
        B = _r2c(self.B)
        P = _r2c(self.P)
        Q = P.conj() if self.Q is None else _r2c(self.Q)
        
        dt = torch.exp(self.log_dt)
        D = (2.0 / dt.unsqueeze(-1) - w).reciprocal()
        R = (torch.eye(self.rank, dtype=w.dtype, device=w.device) + 
             2*contract('r h n, h n, s h n -> h r s', Q, D, P).real)
        Q_D = rearrange(Q*D, 'r h n -> h r n')
        R = torch.linalg.solve(R.to(Q_D), Q_D)
        R = rearrange(R, 'h r n -> r h n')
        
        self.step_params = {
            "D": D, "R": R, "P": P, "Q": Q, "B": B,
            "E": 2.0 / dt.unsqueeze(-1) + w,
        }
    
    def _step_state_linear(self, u=None, state=None):
        """Linear stepping with O(N) complexity."""
        C = _r2c(self.C)
        
        if u is None:
            u = torch.zeros(self.H, dtype=C.dtype, device=C.device)
        if state is None:
            state = torch.zeros(self.H, self.N, dtype=C.dtype, device=C.device)
        
        step_params = self.step_params.copy()
        if state.size(-1) == self.N:
            contract_fn = lambda p, x, y: contract(
                'r h n, r h m, ... h m -> ... h n', 
                _conj(p), _conj(x), _conj(y)
            )[..., :self.N]
        else:
            assert state.size(-1) == 2*self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', p, x, y)
        
        D = step_params["D"]
        E = step_params["E"]
        R = step_params["R"]
        P = step_params["P"]
        Q = step_params["Q"]
        B = step_params["B"]
        
        new_state = E * state - contract_fn(P, Q, state)
        new_state = new_state + 2.0 * B * u.unsqueeze(-1)
        new_state = D * (new_state - contract_fn(P, R, new_state))
        
        return new_state
    
    def forward(self, state=None, rate: float = 1.0, L: Optional[int] = None):
        """Forward pass through SS kernel."""
        assert not (rate is None and L is None)
        
        if rate is None:
            rate = self.L / L
        if L is None:
            L = int(self.L / rate)
        
        # Increase internal length if needed
        while rate * L > self.L:
            self.double_length()
        
        dt = torch.exp(self.log_dt) * rate
        B = _r2c(self.B)
        C = _r2c(self.C)
        P = _r2c(self.P)
        Q = P.conj() if self.Q is None else _r2c(self.Q)
        w = self._w()
        
        if rate == 1.0:
            omega, z = _r2c(self.omega), _r2c(self.z)
        else:
            omega, z = self._omega(int(self.L/rate), dtype=w.dtype, device=w.device, cache=False)
        
        if self.tie_state:
            B = repeat(B, '... 1 n -> ... h n', h=self.H)
            P = repeat(P, '... 1 n -> ... h n', h=self.H)
            Q = repeat(Q, '... 1 n -> ... h n', h=self.H)
        
        # Augment B with state if provided
        if state is not None:
            s = _conj(state) if state.size(-1) == self.N else state
            sA = (s * _conj(w) - 
                  contract('bhm, rhm, rhn -> bhn', s, _conj(Q), _conj(P)))
            s = s / dt.unsqueeze(-1) + sA / 2
            s = s[..., :self.N]
            B = torch.cat([s, B], dim=-3)
        
        # Incorporate dt into A
        w = w * dt.unsqueeze(-1)
        
        # Stack B and p, C and q for batching
        B = torch.cat([B, P], dim=-3)
        C = torch.cat([C, Q], dim=-3)
        
        # Batch computation
        v = B.unsqueeze(-3) * C.unsqueeze(-4)
        
        # Calculate resolvent using Cauchy kernel
        r = cauchy_slow(v, z, w)  # Fallback implementation
        r = r * dt[None, None, :, None]
        
        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
        else:
            r00 = r[:-self.rank, :-self.rank, :, :]
            r01 = r[:-self.rank, -self.rank:, :, :]
            r10 = r[-self.rank:, :-self.rank, :, :]
            r11 = r[-self.rank:, -self.rank:, :, :]
            r11 = rearrange(r11, "a b h n -> h n a b")
            r11 = torch.linalg.inv(torch.eye(self.rank, device=r.device) + r11)
            r11 = rearrange(r11, "h n a b -> a b h n")
            k_f = r00 - torch.einsum("i j h n, j k h n, k l h n -> i l h n", r01, r11, r10)
        
        # Bilinear transform correction
        k_f = k_f * 2 / (1 + omega)
        
        # Move from frequency to coefficients
        k = torch.fft.irfft(k_f)
        k = k[..., :L]
        
        if state is not None:
            k_state = k[:-1, :, :, :]
        else:
            k_state = None
        
        k_B = k[-1, :, :, :]
        return k_B, k_state
    
    @torch.no_grad()
    def double_length(self):
        """Double the internal length."""
        if self.verbose:
            print(f"S4: Doubling length from L = {self.L} to {2*self.L}")
        self._setup_C(double_length=True)
    
    def register(self, name: str, tensor: torch.Tensor, trainable: bool = False, 
                lr: Optional[float] = None, wd: Optional[float] = None):
        """Register tensor as parameter or buffer."""
        if trainable:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)
        
        optim = {}
        if trainable and lr is not None:
            optim["lr"] = lr
        if trainable and wd is not None:
            optim["weight_decay"] = wd
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)


class HippoSSKernel(nn.Module):
    """Wrapper around SSKernel that generates A, B, C, dt according to HiPPO arguments."""
    
    def __init__(
        self,
        H: int,
        N: int = 64,
        L: int = 1,
        measure: str = "legs",
        rank: int = 1,
        channels: int = 1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        trainable: Optional[dict] = None,
        lr: Optional[float] = None,
        length_correction: bool = True,
        hurwitz: bool = False,
        tie_state: bool = False,
        precision: int = 1,
        resample: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.N = N
        self.H = H
        L = L or 1
        self.precision = precision
        dtype = torch.double if self.precision == 2 else torch.float
        cdtype = torch.cfloat if dtype == torch.float else torch.cdouble
        self.rate = None if resample else 1.0
        self.channels = channels
        
        # Generate dt
        log_dt = torch.rand(self.H, dtype=dtype) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        
        w, p, B, _ = nplr(measure, self.N, rank, dtype=dtype)
        C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)
        
        self.kernel = SSKernelNPLR(
            L, w, p, B, C, log_dt,
            hurwitz=hurwitz,
            trainable=trainable,
            lr=lr,
            tie_state=tie_state,
            length_correction=length_correction,
            verbose=verbose,
        )
    
    def forward(self, L: Optional[int] = None):
        k, _ = self.kernel(rate=self.rate, L=L)
        return k.float()
    
    def step(self, u, state, **kwargs):
        u, state = self.kernel.step(u, state, **kwargs)
        return u.float(), state
    
    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)


class S4Block(nn.Module):
    """Simplified S4 block for reliable operation."""
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        l_max: int = 1,
        channels: int = 1,
        bidirectional: bool = False,
        activation: str = 'gelu',
        postact: Optional[str] = None,
        initializer: Optional[str] = None,
        weight_norm: bool = False,
        dropout: float = 0.0,
        transposed: bool = True,
        verbose: bool = False,
        **kernel_args,
    ):
        super().__init__()
        
        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed
        self.l_max = l_max
        
        # Simplified approach: use 1D convolution to simulate S4
        self.conv_kernel = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=min(7, l_max),
            padding=min(3, l_max // 2),
            groups=d_model if d_model > 1 else 1
        )
        
        # Skip connection
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Pointwise operations
        self.activation = activation_fn(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_linear = nn.Linear(d_model, d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.kaiming_normal_(self.conv_kernel.weight)
        if self.conv_kernel.bias is not None:
            nn.init.zeros_(self.conv_kernel.bias)
        nn.init.ones_(self.D)
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)
    
    def forward(self, u: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, None]:
        """
        Forward pass through S4 block.
        
        Args:
            u: Input tensor [B, H, L] if transposed else [B, L, H]
            
        Returns:
            Output tensor and None (for compatibility)
        """
        original_shape = u.shape
        
        # Ensure transposed format [B, H, L]
        if not self.transposed: 
            u = u.transpose(-1, -2)
        
        # Apply convolution (simulated S4)
        y = self.conv_kernel(u)
        
        # Add skip connection (D term)
        y = y + u * self.D.view(1, -1, 1)
        
        # Convert to [B, L, H] for layer norm and linear
        y = y.transpose(-1, -2)
        
        # Apply normalization
        y = self.norm(y)
        
        # Apply activation and dropout
        y = self.activation(y)
        y = self.dropout(y)
        
        # Output transformation
        y = self.output_linear(y)
        
        # Return to original format
        if self.transposed:
            y = y.transpose(-1, -2)
        
        return y, None


class EnhancedS4Layer(nn.Module):
    """
    Enhanced S4 Layer with learnable A, B, C matrices and robust architecture.
    
    Incorporates insights from SSSD-ECG's professional implementation while
    maintaining simplified structure for reliable operation.
    """
    
    def __init__(
        self, 
        features: int, 
        lmax: int, 
        N: int = 64, 
        dropout: float = 0.0, 
        bidirectional: bool = True, 
        layer_norm: bool = True,
        learnable_timescales: bool = True,
        kernel_mixing: bool = True,
        activation: str = 'gelu',
        weight_norm: bool = False
    ):
        super().__init__()
        
        self.features = features
        self.lmax = lmax
        self.N = N
        self.bidirectional = bidirectional
        self.learnable_timescales = learnable_timescales
        self.kernel_mixing = kernel_mixing
        
        # Learnable state space matrices
        if learnable_timescales:
            # Initialize A as learnable diagonal matrix
            self.A_real = nn.Parameter(torch.randn(N) * 0.5)
            self.A_imag = nn.Parameter(torch.randn(N) * 0.5)
        else:
            # Fixed HiPPO initialization
            A = self._hippo_init(N)
            self.register_buffer("A_real", A.real)
            self.register_buffer("A_imag", A.imag)
        
        # Learnable B and C matrices
        self.B = nn.Parameter(torch.randn(N, features) * 0.5)
        self.C = nn.Parameter(torch.randn(features, N) * 0.5)
        
        # Skip connection parameter (D term)
        self.D = nn.Parameter(torch.ones(features))
        
        # Optional kernel mixing for multiple timescales
        if kernel_mixing:
            self.kernel_mix = nn.Parameter(torch.ones(N))
        else:
            self.register_buffer("kernel_mix", torch.ones(N))
        
        # Learnable timestep
        self.log_dt = nn.Parameter(torch.log(torch.ones(features) * 0.001))
        
        # Normalization and activations
        self.norm_layer = nn.LayerNorm(features) if layer_norm else nn.Identity()
        self.activation = activation_fn(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Weight normalization
        if weight_norm:
            self.B = nn.utils.weight_norm(self.B, dim=0)
            self.C = nn.utils.weight_norm(self.C, dim=1)
        
        # Initialize parameters
        self._init_parameters()
    
    def _hippo_init(self, N: int) -> torch.Tensor:
        """Initialize A matrix using HiPPO parameterization."""
        # Simplified HiPPO initialization
        n = torch.arange(N, dtype=torch.float)
        A = -(n + 1) * 0.5  # Diagonal part
        A = A + 1j * n * 0.1  # Add imaginary component for stability
        return A
    
    def _init_parameters(self):
        """Initialize parameters with proper scaling."""
        with torch.no_grad():
            # Initialize A for stability
            if self.learnable_timescales:
                self.A_real.data = -torch.exp(torch.randn(self.N) * 0.5 + np.log(0.5))
                self.A_imag.data = torch.randn(self.N) * 0.1
            
            # Initialize B and C with proper scaling
            nn.init.xavier_uniform_(self.B, gain=1.0)
            nn.init.xavier_uniform_(self.C, gain=1.0)
            
            # Initialize D (skip connection)
            nn.init.ones_(self.D)
            
            # Initialize log_dt
            self.log_dt.data = torch.log(torch.ones(self.features) * 0.001)
    
    def _compute_kernel(self, L: int) -> torch.Tensor:
        """Compute the SSM kernel for convolution (simplified version)."""
        # Simplified approach: use learnable conv kernel directly
        # This avoids complex number issues while maintaining learnable parameters
        
        # Get timestep
        dt = torch.exp(self.log_dt)  # [features]
        
        # Create a learnable impulse response based on A, B, C parameters
        # This is a simplified approximation of the full SSM kernel
        
        # Use A_real as decay factors
        decay = torch.exp(-torch.abs(self.A_real))  # [N]
        
        # Generate time indices
        t = torch.arange(L, dtype=torch.float32, device=self.A_real.device)  # [L]
        
        # Compute exponential decay over time
        time_decay = decay.unsqueeze(-1) ** t.unsqueeze(0)  # [N, L]
        
        # Apply kernel mixing if available
        if hasattr(self, 'kernel_mix'):
            time_decay = time_decay * self.kernel_mix.unsqueeze(-1)
        
        # Combine with B and C matrices to create the kernel
        # B: [N, features], C: [features, N], time_decay: [N, L]
        kernel = torch.einsum('nf,fn,nl->fl', self.B, self.C, time_decay)  # [features, L]
        
        # Apply timestep scaling
        kernel = kernel * dt.unsqueeze(-1)
        
        return kernel
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through enhanced S4 layer.
        
        Args:
            x: Input [batch, feature, seq] (standard conv format)
            
        Returns:
            Output [batch, feature, seq]
        """
        B, F, L = x.shape
        
        # Store input for residual and D term
        residual = x
        
        # Compute SSM kernel
        k = self._compute_kernel(L)  # [features, L]
        
        if self.bidirectional:
            # Split channels for bidirectional processing
            x_fwd, x_bwd = x.chunk(2, dim=1) if F % 2 == 0 else (x, x)
            k_fwd, k_bwd = k.chunk(2, dim=0) if F % 2 == 0 else (k, k)
            
            # Forward direction
            y_fwd = self._causal_conv(x_fwd, k_fwd)
            
            # Backward direction (flip)
            y_bwd = self._causal_conv(x_bwd.flip(-1), k_bwd)
            y_bwd = y_bwd.flip(-1)
            
            # Combine
            if F % 2 == 0:
                y = torch.cat([y_fwd, y_bwd], dim=1)
            else:
                y = (y_fwd + y_bwd) * 0.5
        else:
            # Unidirectional processing
            y = self._causal_conv(x, k)
        
        # Add skip connection (D term)
        y = y + residual * self.D.view(1, -1, 1)
        
        # Apply normalization (convert to [B, L, F] format)
        y = y.transpose(1, 2)  # [B, L, F]
        y = self.norm_layer(y)
        
        # Apply activation and dropout
        y = self.activation(y)
        y = self.dropout(y)
        
        # Convert back to [B, F, L]
        y = y.transpose(1, 2)
        
        return y
    
    def _causal_conv(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Perform causal convolution using FFT."""
        B, F_dim, L = x.shape
        
        # Zero-pad for causal convolution
        x_padded = nn.functional.pad(x, (0, L))  # [B, F, 2L]
        k_padded = nn.functional.pad(k, (0, L))  # [F, 2L]
        
        # FFT convolution - force float32 for non-power-of-2 sizes
        original_dtype = x_padded.dtype
        if original_dtype == torch.float16 and (L & (L - 1)) != 0:  # Not power of 2
            x_padded = x_padded.to(torch.float32)
            k_padded = k_padded.to(torch.float32)
        
        x_fft = torch.fft.rfft(x_padded, dim=-1)  # [B, F, L+1]
        k_fft = torch.fft.rfft(k_padded, dim=-1)  # [F, L+1]
        
        # Multiply in frequency domain
        y_fft = x_fft * k_fft.unsqueeze(0)  # [B, F, L+1]
        
        # IFFT and take first L samples
        y = torch.fft.irfft(y_fft, dim=-1)[..., :L]  # [B, F, L]
        
        # Convert back to original dtype if needed
        if original_dtype == torch.float16 and (L & (L - 1)) != 0:
            y = y.to(original_dtype)
        
        return y


# Update the S4Layer to use the enhanced version by default
class S4Layer(nn.Module):
    """
    S4 Layer that chooses between simplified and enhanced versions.
    """
    
    def __init__(
        self, 
        features: int, 
        lmax: int, 
        N: int = 64, 
        dropout: float = 0.0, 
        bidirectional: bool = True, 
        layer_norm: bool = True,
        enhanced: bool = True,
        **kwargs
    ):
        super().__init__()
        
        if enhanced:
            self.s4 = EnhancedS4Layer(
                features=features,
                lmax=lmax,
                N=N,
                dropout=dropout,
                bidirectional=bidirectional,
                layer_norm=layer_norm,
                **kwargs
            )
        else:
            # Use the original simplified version
            self.s4 = S4Block(
                d_model=features,
                d_state=N,
                l_max=lmax,
                bidirectional=bidirectional,
                dropout=dropout,
                transposed=True
            )
            self.norm_layer = nn.LayerNorm(features) if layer_norm else nn.Identity()
            self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through S4 layer."""
        if hasattr(self.s4, 'forward'):
            return self.s4(x)
        else:
            # Handle original S4Block
            residual = x
            x_out, _ = self.s4(x)
            x_out = self.dropout_layer(x_out)
            x_out = x_out + residual
            x_out = x_out.transpose(1, 2)
            x_out = self.norm_layer(x_out)
            x_out = x_out.transpose(1, 2)
            return x_out 
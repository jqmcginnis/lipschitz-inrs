import numpy as np

def _scale_to_product(s_raw, k):
    """Scale positive s_raw multiplicatively so that product equals k."""
    s_raw = np.asarray(s_raw, dtype=float)
    assert np.all(s_raw > 0), "All s_raw must be positive."
    L = s_raw.size
    c = (k / np.prod(s_raw)) ** (1.0 / L)
    return s_raw * c

def _allocate_with_last_and_product(k, r, f):
    """
    Given a shape vector f (length L), find u = a + b*f such that:
      (i) u_{L-1} = log(r)
     (ii) sum(u) = log(k)
    Then s = exp(u).
    """
    f = np.asarray(f, dtype=float)
    L = f.size
    assert L >= 1
    if L == 1:
        s = np.array([float(k)], dtype=float)
        return s, np.log(s)

    A = np.log(r)
    f_last = f[-1]
    sum_f = f.sum()

    denom = (sum_f - L * f_last)
    # If denom is (near) zero, fall back to simple fix: set all equal except last fixed, then rescale front L-1 via a constant.
    if np.isclose(denom, 0.0):
        # Put u_last = A, others equal: u_i = x for i < L-1; choose x so sum u = log k.
        x = (np.log(k) - A) / (L - 1)
        u = np.full(L, x, dtype=float)
        u[-1] = A
        s = np.exp(u)
        return s, u

    b = (np.log(k) - L * A) / denom
    a = A - b * f_last
    u = a + b * f
    s = np.exp(u)
    return s, u

def allocate_first(k, L, r=1.0):
    """All budget in first layer."""
    s = np.ones(L, dtype=float)
    s[0] = k
    u = np.log(s)
    return s, u

def allocate_uniform(k, L, r=1.0):
    """Uniform allocation in logK space (equivalently uniform in budget space under product constraint)."""
    logk = np.log(k)
    u = np.full(L, logk / L, dtype=float)
    s = np.exp(u)
    return s, u

def allocate_exponential(k, L, r=1.0):
    """
    Linear profile in log-space (=> exponential in budget space),
    with the LAST element fixed to r and product ∏ s_i = k.
    """
    assert L >= 1
    if L == 1:
        s = np.array([float(k)], dtype=float)
        return s, np.log(s)

    # v: front-heavy ramp in log-space (front=1 -> back=0)
    v = np.linspace(1.0, 0.0, L)
    sum_v = v.sum()

    # u = a + b*v, with:
    #   u[-1] = log(r)  (since v[-1] = 0 -> a = log(r))
    #   sum(u) = log(k)        -> L*a + b*sum(v) = log(k)
    a = np.log(r)
    b = (np.log(k) - L * a) / sum_v

    u = a + b * v
    s = np.exp(u)
    return s, u


def allocate_linear(k, L, r, eps=1e-12):
    # linearly increasing backward: s[0] ... s[L-2] on a line to s[L-1]=r
    assert L >= 1 and k > 0 and r > 0
    if L == 1:
        s = np.array([float(k)]); return s, np.log(s)
    if L == 2:
        s = np.array([float(k)/float(r), float(r)]); return s, np.log(s)

    t = np.linspace(0.0, 1.0, L)                    # 0...1
    logk = np.log(k)

    def logprod(s0):
        s = s0 + (r - s0) * t               # linear from s0 → r
        return np.sum(np.log(np.maximum(s, eps)))   # avoid log(0)

    lo, hi = eps, max(r, k**(1.0/L)) * 2.0   # bracket s0
    while logprod(hi) < logk: hi *= 2.0             # expand if needed
    for _ in range(50):                              # quick bisection
        mid = 0.5*(lo+hi)
        (lo if logprod(mid) < logk else hi).__class__  # no-op to keep it tight
        if logprod(mid) < logk: lo = mid
        else: hi = mid

    s0 = 0.5*(lo+hi)
    s = s0 + (r - s0) * t
    return s, np.log(s)


def allocate_cosine_annealed(k, L, r, variant="front", tol=1e-12, max_it=80):
    """
    Cosine-annealed allocation *in budget space* with last entry fixed to r.
    Solves for the cosine amplitude alpha so that ∏ K_i = k exactly.

    Args:
        k   : total multiplicative budget (>0)
        L     : number of components (>=2)
        r : fixed last component budget (>0), i.e., K[L-1] = r
        variant: "front" -> g(t) = (1+cos(pi t))/2 ; "mid" -> (1 - cos(2 pi t))/2
        tol   : bisection tolerance in log-domain
        max_it: max iterations

    Returns:
        K : (L,) per-component budgets (product == k, last == r)
        u : (L,) log-budgets
        alpha : solved cosine amplitude
    """
    assert k > 0 and r > 0 and L >= 2
    t = np.linspace(0.0, 1.0, L)           # t[0]=0 ... t[L-1]=1
    if variant == "front":
        g = 0.5 * (1.0 + np.cos(np.pi * t))
    elif variant == "mid":
        g = 0.5 * (1.0 - np.cos(2.0 * np.pi * t))
    else:
        raise ValueError("variant must be 'front' or 'mid'.")

    # Last must be exactly r, so enforce K[L-1] = r -> g[L-1] not used (it is 0 for 'front').
    g_front = g[:-1]                # use first L-1 entries
    target = np.log(k / (r**L))

    # F(alpha) = sum log(1 + alpha g_i) - target = 0, with alpha > -1/max(g_i_nonzero)
    g_pos = g_front[g_front > 0]
    if g_pos.size == 0:
        # degenerate: no modulation -> product is r^L; must have k == r^L
        if not np.isclose(k, r**L):
            raise ValueError("No cosine leverage (all g=0) but k != r^L.")
        K = np.full(L, r, dtype=float)
        u = np.log(K)
        return K, u, 0.0

    alpha_lo = -1.0 / np.max(g_pos) + 1e-15      # just above feasibility boundary
    alpha_hi = 1.0
    # Expand upper bound until F(alpha_hi) >= 0 (monotone increasing in alpha for g>=0)
    def F(a):
        return np.sum(np.log1p(a * g_front)) - target
    while F(alpha_hi) < 0.0:
        alpha_hi *= 2.0
        if alpha_hi > 1e12:
            raise RuntimeError("alpha_hi exploded; check k / r^L is not too large.")

    # Bisection
    for _ in range(max_it):
        alpha_mid = 0.5 * (alpha_lo + alpha_hi)
        val = F(alpha_mid)
        if abs(val) < tol:
            alpha = alpha_mid
            break
        if val < 0:
            alpha_lo = alpha_mid
        else:
            alpha_hi = alpha_mid
    else:
        alpha = 0.5 * (alpha_lo + alpha_hi)

    # Construct budgets
    K = np.empty(L, dtype=float)
    K[:-1] = r * (1.0 + alpha * g_front)
    K[-1]  = r
    u = np.log(K)
    return K, u


def get_allocation_function(name):
    name = name.lower()
    if name in ["uniform", "uni"]:
        return allocate_uniform
    elif name in ["allfirst", "first"]:
        return allocate_first
    elif name in ["exponential", "exp"]:
        return allocate_exponential
    elif name in ["linear", "lin"]:
        return allocate_linear
    elif name in ["cosine", "cos"]:
        return allocate_cosine_annealed
    else:
        raise ValueError(f"Unknown allocation function name: {name}")

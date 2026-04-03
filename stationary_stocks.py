from __future__ import annotations
import numpy as np

DEFAULT_EPSILON = 1e-12

def check_policy_feasibility(Gamma, P, d, tr, n, tol=1e-10):
    """
    Validate shapes, nonnegativity, probability constraints, and basic
    consistency conditions needed by the stock solver.

    Traveler mass is normalized to 1, so total cab mass is n.
    """
    Gamma = np.asarray(Gamma, dtype=float)
    P = np.asarray(P, dtype=float)
    d = np.asarray(d, dtype=float)
    tr = np.asarray(tr, dtype=float)

    if Gamma.ndim != 3:
        raise ValueError(f"Gamma must be 3D with shape (L, L, L), got {Gamma.shape}.")

    L = Gamma.shape[0]
    if Gamma.shape != (L, L, L):
        raise ValueError(f"Gamma must have shape (L, L, L), got {Gamma.shape}.")
    if P.shape != (L, L):
        raise ValueError(f"P must have shape ({L}, {L}), got {P.shape}.")
    if d.shape != (L, L):
        raise ValueError(f"d must have shape ({L}, {L}), got {d.shape}.")
    if tr.shape != (L, L):
        raise ValueError(f"tr must have shape ({L}, {L}), got {tr.shape}.")

    for name, array in [("Gamma", Gamma), ("P", P), ("d", d), ("tr", tr)]:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or Inf values.")

    if np.any(Gamma < -tol):
        raise ValueError("Gamma must be nonnegative.")
    if np.any(P < -tol):
        raise ValueError("P must be nonnegative.")
    if np.any(d < -tol):
        raise ValueError("d must be nonnegative.")
    if np.any(tr < -tol):
        raise ValueError("tr must be nonnegative.")
    if np.any(d > 1.0 + tol):
        raise ValueError("d must lie in [0, 1] up to tolerance.")
    if n <= 0:
        raise ValueError("n must be strictly positive.")

    gamma_row_sums = Gamma.sum(axis=(1, 2))
    p_row_sums = P.sum(axis=1)

    if not np.allclose(gamma_row_sums, 1.0, atol=tol, rtol=0.0):
        raise ValueError(
            "For each origin k, Gamma[k, :, :].sum() must equal 1 up to tolerance."
        )
    if not np.allclose(p_row_sums, 1.0, atol=tol, rtol=0.0):
        raise ValueError(
            "For each origin i, P[i, :].sum() must equal 1 up to tolerance."
        )

    zero_tr_mask = tr <= tol
    if np.any(zero_tr_mask):
        for i in range(L):
            for j in range(L):
                if zero_tr_mask[i, j] and np.any(Gamma[:, i, j] > tol) and i != j:
                    raise ValueError(
                        "tr[i, j] is zero but Gamma[:, i, j] assigns positive mass "
                        f"to traveler type ({i}, {j})."
                    )

    return {
        "Gamma": np.clip(Gamma, 0.0, None),
        "P": np.clip(P, 0.0, None),
        "d": np.clip(d, 0.0, 1.0),
        "tr": np.clip(tr, 0.0, None),
        "L": L,
    }
    
def compute_theta(vNH_loc, Gamma, tr, n, epsilon=DEFAULT_EPSILON):
    """
    Compute theta[k, i, j] = n * vNH_loc[k] * Gamma[k, i, j] / tr[i, j].

    Special convention:
    - If tr[i, j] > epsilon, use the standard formula.
    - If tr[i, j] <= epsilon and numerator <= epsilon, set theta = 0.
    - If tr[i, j] <= epsilon and k == i == j with positive numerator,
      set theta = -1 as a marker for the special inactive case.
    - Otherwise raise an error.
    """
    vNH_loc = np.asarray(vNH_loc, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)
    tr = np.asarray(tr, dtype=float)

    L = Gamma.shape[0]
    if vNH_loc.shape != (L,):
        raise ValueError(f"vNH_loc must have shape ({L},), got {vNH_loc.shape}.")

    theta = np.zeros((L, L, L), dtype=float)

    for k in range(L):
        for i in range(L):
            for j in range(L):
                numerator = n * vNH_loc[k] * Gamma[k, i, j]

                if tr[i, j] > epsilon:
                    theta[k, i, j] = numerator / tr[i, j]
                else:
                    if numerator <= epsilon:
                        theta[k, i, j] = -2.0
                    elif k == i == j:
                        theta[k, i, j] = -1.0
                    else:
                        raise ValueError(
                            "Encountered positive applicant mass for a traveler type "
                            f"with tr[{i}, {j}] = 0 at indices (k, i, j)=({k}, {i}, {j})."
                        )

    if not np.all(np.isfinite(theta)):
        raise ValueError("theta contains NaN or Inf values.")

    return theta

def compute_theta_less(theta, d):
    """
    ThetaLess[k, i, j] = sum_{h : d[h, i] < d[k, i]} theta[h, i, j]

    The ordering is implemented exactly as requested, using the strict
    comparison d[h, i] < d[k, i].
    """
    theta = np.asarray(theta, dtype=float)
    d = np.asarray(d, dtype=float)
    L = theta.shape[0]
    if theta.shape != (L, L, L):
        raise ValueError(f"theta must have shape (L, L, L), got {theta.shape}.")
    if d.shape != (L, L):
        raise ValueError(f"d must have shape ({L}, {L}), got {d.shape}.")
    ThetaLess = np.zeros_like(theta)
    for k in range(L):
        for i in range(L):
            closer_mask = d[:, i] < d[k, i]
            for j in range(L):
                ThetaLess[k, i, j] = theta[closer_mask, i, j].sum()
    if not np.all(np.isfinite(ThetaLess)):
        raise ValueError("ThetaLess contains NaN or Inf values.")
    return ThetaLess

def compute_pi_trav(theta, ThetaLess):
    """
    pi_trav[k, i, j] = exp(-ThetaLess[k, i, j]) * (1 - exp(-theta[k, i, j]))
    """
    theta = np.asarray(theta, dtype=float)
    ThetaLess = np.asarray(ThetaLess, dtype=float)
    if theta.shape != ThetaLess.shape:
        raise ValueError(
            f"theta and ThetaLess must have the same shape, got {theta.shape} and "
            f"{ThetaLess.shape}."
        )
    pi_trav = np.exp(-ThetaLess) * (1.0 - np.exp(-theta))
    if not np.all(np.isfinite(pi_trav)):
        raise ValueError("pi_trav contains NaN or Inf values.")
    return np.clip(pi_trav, 0.0, 1.0)

def compute_pi_cab(theta, ThetaLess, epsilon=DEFAULT_EPSILON):
    """
    Compute pi_cab from theta and ThetaLess.

    Conventions:
    - theta == -1: special diagonal inactive case, force pi_cab = 1
    - theta == -2: nonexistent market / zero assigned mass, force pi_cab = 0
    - otherwise: use the standard formula
          pi_cab = exp(-ThetaLess) * (1 - exp(-theta)) / theta
      with limiting value exp(-ThetaLess) when theta is numerically zero.
    """
    theta = np.asarray(theta, dtype=float)
    ThetaLess = np.asarray(ThetaLess, dtype=float)

    if theta.shape != ThetaLess.shape:
        raise ValueError(
            f"theta and ThetaLess must have the same shape, got {theta.shape} and "
            f"{ThetaLess.shape}."
        )

    pi_cab = np.zeros_like(theta)

    inactive_mask = theta == -1.0
    empty_market_mask = theta == -2.0
    regular_mask = ~(inactive_mask | empty_market_mask)

    nonzero_mask = regular_mask & (np.abs(theta) > epsilon)
    zero_mask = regular_mask & (np.abs(theta) <= epsilon)

    pi_cab[nonzero_mask] = (
        np.exp(-ThetaLess[nonzero_mask])
        * (1.0 - np.exp(-theta[nonzero_mask]))
        / theta[nonzero_mask]
    )

    pi_cab[zero_mask] = np.exp(-ThetaLess[zero_mask])

    pi_cab[inactive_mask] = .0
    pi_cab[empty_market_mask] = 0.0

    if not np.all(np.isfinite(pi_cab)):
        raise ValueError("pi_cab contains NaN or Inf values.")

    lower_tol = -1e-10
    upper_tol = 1.0 + 1e-10
    if np.any(pi_cab < lower_tol) or np.any(pi_cab > upper_tol):
        raise ValueError("pi_cab must lie in [0, 1] up to numerical tolerance.")

    return np.clip(pi_cab, 0.0, 1.0)
  
def update_stocks_given_pi(vNH_loc, Gamma, P, d, pi_cab, tol=1e-10):
    """
    Update the stationary stock blocks holding the current vNH_loc fixed when
    computing pi_cab.
    """
    vNH_loc = np.asarray(vNH_loc, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)
    P = np.asarray(P, dtype=float)
    d = np.asarray(d, dtype=float)
    pi_cab = np.asarray(pi_cab, dtype=float)

    L = Gamma.shape[0]

    if vNH_loc.shape != (L,):
        raise ValueError(f"vNH_loc must have shape ({L},), got {vNH_loc.shape}.")
    if P.shape != (L, L):
        raise ValueError(f"P must have shape ({L}, {L}), got {P.shape}.")
    if d.shape != (L, L):
        raise ValueError(f"d must have shape ({L}, {L}), got {d.shape}.")
    if pi_cab.shape != (L, L, L):
        raise ValueError(f"pi_cab must have shape ({L}, {L}, {L}), got {pi_cab.shape}.")

    unmatched_mass = np.zeros(L, dtype=float)
    for i in range(L):
        unmatched_probability = 0.0
        for a in range(L):
            for b in range(L):
                unmatched_probability += Gamma[i, a, b] * (1.0 - pi_cab[i, a, b])
        unmatched_mass[i] = vNH_loc[i] * unmatched_probability

    vH_trans = np.zeros((L, L, L), dtype=float)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                inflow = vNH_loc[i] * Gamma[i, j, k] * pi_cab[i, j, k]
                if d[i, j] > tol:
                    vH_trans[i, j, k] = inflow / d[i, j]
                else:
                    if inflow > tol:
                        raise ValueError(
                            "d[i, j] = 0 but hailed-transit inflow is positive at "
                            f"indices ({i}, {j}, {k})."
                        )
                    vH_trans[i, j, k] = 0.0

    vH_loc = np.zeros((L, L), dtype=float)
    for i in range(L):
        for j in range(L):
            total = 0.0
            for k in range(L):
                total += d[k, i] * vH_trans[k, i, j]
            vH_loc[i, j] = total

    vNH_trans = np.zeros((L, L), dtype=float)
    stay_put_mass = np.zeros(L, dtype=float)

    for i in range(L):
        stay_put_mass[i] = unmatched_mass[i] * P[i, i]

        for j in range(L):
            if j == i:
                vNH_trans[i, i] = 0.0
            else:
                inflow = (
                    unmatched_mass[i] * P[i, j]
                    + vNH_loc[i] * Gamma[i, i, j] * pi_cab[i, i, j]
                    + vH_loc[i, j]
                )

                if d[i, j] > tol:
                    vNH_trans[i, j] = inflow / d[i, j]
                else:
                    if inflow > tol:
                        raise ValueError(
                            "d[i, j] = 0 but not-hailed-transit inflow is positive at "
                            f"indices ({i}, {j})."
                        )
                    vNH_trans[i, j] = 0.0

    vNH_loc_next = np.zeros(L, dtype=float)
    for i in range(L):
        vNH_loc_next[i] = stay_put_mass[i]
        for j in range(L):
            if j != i:
                vNH_loc_next[i] += d[j, i] * vNH_trans[j, i]

    for name, array in [
        ("unmatched_mass", unmatched_mass),
        ("vH_trans", vH_trans),
        ("vH_loc", vH_loc),
        ("vNH_trans", vNH_trans),
        ("vNH_loc", vNH_loc_next),
    ]:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or Inf values.")
        if np.any(array < -tol):
            raise ValueError(f"{name} contains negative values beyond tolerance.")

    return {
        "unmatched_mass": np.clip(unmatched_mass, 0.0, None),
        "vNH_loc": np.clip(vNH_loc_next, 0.0, None),
        "vNH_trans": np.clip(vNH_trans, 0.0, None),
        "vH_loc": np.clip(vH_loc, 0.0, None),
        "vH_trans": np.clip(vH_trans, 0.0, None),
    }
    
def compute_residuals(vNH_loc, vNH_trans, vH_loc, vH_trans, Gamma, P, d, pi_cab, n):
    """
    Compute residuals of each stationary stock equation and the total-mass condition.
    Since traveler mass is 1, the total cab mass must equal n.
    """
    vNH_loc = np.asarray(vNH_loc, dtype=float)
    vNH_trans = np.asarray(vNH_trans, dtype=float)
    vH_loc = np.asarray(vH_loc, dtype=float)
    vH_trans = np.asarray(vH_trans, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)
    P = np.asarray(P, dtype=float)
    d = np.asarray(d, dtype=float)
    pi_cab = np.asarray(pi_cab, dtype=float)

    L = vNH_loc.shape[0]

    unmatched_mass = np.zeros(L, dtype=float)
    for i in range(L):
        unmatched_probability = 0.0
        for a in range(L):
            for b in range(L):
                unmatched_probability += Gamma[i, a, b] * (1.0 - pi_cab[i, a, b])
        unmatched_mass[i] = vNH_loc[i] * unmatched_probability

    nh_trans_residual = np.zeros((L, L), dtype=float)
    stay_put_mass = np.zeros(L, dtype=float)

    for i in range(L):
      stay_put_mass[i] = unmatched_mass[i] * P[i, i]
      for j in range(L):
        if j == i:
            nh_trans_residual[i, i] = vNH_trans[i, i]  # should be zero
        else:
            inflow = (
                unmatched_mass[i] * P[i, j]
                + vNH_loc[i] * Gamma[i, i, j] * pi_cab[i, i, j]
                + vH_loc[i, j]
            )
            nh_trans_residual[i, j] = d[i, j] * vNH_trans[i, j] - inflow

    nh_loc_residual = np.zeros(L, dtype=float)
    for i in range(L):
      rhs = stay_put_mass[i]
      for j in range(L):
          if j != i:
            rhs += d[j, i] * vNH_trans[j, i]
      nh_loc_residual[i] = vNH_loc[i] - rhs
      
    h_trans_residual = np.zeros((L, L, L), dtype=float)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                rhs = vNH_loc[i] * Gamma[i, j, k] * pi_cab[i, j, k]
                h_trans_residual[i, j, k] = d[i, j] * vH_trans[i, j, k] - rhs

    h_loc_residual = np.zeros((L, L), dtype=float)
    for i in range(L):
        for j in range(L):
            rhs = 0.0
            for k in range(L):
                rhs += d[k, i] * vH_trans[k, i, j]
            h_loc_residual[i, j] = vH_loc[i, j] - rhs

    total_mass = (
        vNH_loc.sum()
        + vNH_trans.sum()
        + vH_loc.sum()
        + vH_trans.sum()
    )
    mass_residual = total_mass - 1.0

    max_abs = {
        "vNH_trans": float(np.max(np.abs(nh_trans_residual))),
        "vNH_loc": float(np.max(np.abs(nh_loc_residual))),
        "vH_trans": float(np.max(np.abs(h_trans_residual))),
        "vH_loc": float(np.max(np.abs(h_loc_residual))),
        "mass": float(abs(mass_residual)),
    }
    max_abs["overall"] = max(max_abs.values())

    return {
        "vNH_trans": nh_trans_residual,
        "vNH_loc": nh_loc_residual,
        "vH_trans": h_trans_residual,
        "vH_loc": h_loc_residual,
        "mass": mass_residual,
        "max_abs": max_abs,
    }

def solve_stationary_stocks(Gamma,P,d,tr,n,tol=1e-10,max_iter=10000,damp=0.2):
    """
    Solve the stationary stock fixed point s = T(s; Gamma, P).

    Traveler mass is normalized to 1, so the total cab mass is n.
    """
    if not (0.0 < damp <= 1.0):
        raise ValueError("damp must lie in (0, 1].")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")

    checked = check_policy_feasibility(Gamma, P, d, tr, n, tol=tol)
    Gamma = checked["Gamma"]
    P = checked["P"]
    d = checked["d"]
    tr = checked["tr"]
    L = checked["L"]

    # Initial guess: all cabs not hailed and evenly split across locations.
    # Total cab mass is n.
    vNH_loc = np.full(L, 1.0 / L, dtype=float)
    vNH_trans = np.zeros((L, L), dtype=float)
    vH_loc = np.zeros((L, L), dtype=float)
    vH_trans = np.zeros((L, L, L), dtype=float)

    converged = False
    iterations = 0

    for iterations in range(1, max_iter + 1):
        theta = compute_theta(vNH_loc, Gamma, tr, n)
        ThetaLess = compute_theta_less(theta, d)
        pi_cab = compute_pi_cab(theta, ThetaLess)

        candidate = update_stocks_given_pi(vNH_loc, Gamma, P, d, pi_cab, tol=tol)

        vNH_loc_next = candidate["vNH_loc"]
        vNH_trans_next = candidate["vNH_trans"]
        vH_loc_next = candidate["vH_loc"]
        vH_trans_next = candidate["vH_trans"]
        
        total_mass_next = (
          vNH_loc_next.sum()
          + vNH_trans_next.sum()
          + vH_loc_next.sum()
          + vH_trans_next.sum()
        )

        if total_mass_next <= tol:
          raise ValueError("Total mass collapsed to zero.")

        scale = 1.0 / total_mass_next

        vNH_loc_next *= scale
        vNH_trans_next *= scale
        vH_loc_next *= scale
        vH_trans_next *= scale

        diff = max(
            float(np.max(np.abs(vNH_loc_next - vNH_loc))),
            float(np.max(np.abs(vNH_trans_next - vNH_trans))),
            float(np.max(np.abs(vH_loc_next - vH_loc))),
            float(np.max(np.abs(vH_trans_next - vH_trans))),
        )

        vNH_loc = vNH_loc_next
        vNH_trans = vNH_trans_next
        vH_loc = vH_loc_next
        vH_trans = vH_trans_next

        if diff < tol:
            converged = True
            break

    theta = compute_theta(vNH_loc, Gamma, tr, n)
    ThetaLess = compute_theta_less(theta, d)
    pi_cab = compute_pi_cab(theta, ThetaLess)
    residuals = compute_residuals(
        vNH_loc, vNH_trans, vH_loc, vH_trans, Gamma, P, d, pi_cab, n
    )

    return {
        "vNH_loc": vNH_loc,
        "vNH_trans": vNH_trans,
        "vH_loc": vH_loc,
        "vH_trans": vH_trans,
        "theta": theta,
        "ThetaLess": ThetaLess,
        "pi_cab": pi_cab,
        "residuals": residuals,
        "iterations": iterations,
        "converged": converged,
    }

if "res" in globals():
    del res
    
# Number of locations
L = 3

# Traveler mass = 1, so total cab mass = n
n = 1

# Travel-completion probabilities
d = np.array([
    [1.0, 0.5, 0.5],
    [0.5, 1.0, 0.5],
    [0.5, 0.5, 1.0],
])

# Traveler masses by type (i,j)
# Put zero on the diagonal so your special theta markers can matter there.
tr = np.array([
    [0.0, 0.1666, 0.1666],
    [0.1666, 0.0, 0.1666],
    [0.1666, 0.167, 0.0],
])

# Relocation policy: stay put
P = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])

Gamma = np.zeros((L, L, L))

for k in range(L):
    for i in range(L):
        for j in range(L):
            if i != j:
                Gamma[k, i, j] = 1.0 / 6.0
                

res = solve_stationary_stocks(
    Gamma=Gamma,
    P=P,
    d=d,
    tr=tr,
    n=n,
    tol=1e-10,
    max_iter=10000,
    damp=0.2,
)

print("Converged:", res["converged"])
print("Iterations:", res["iterations"])
print("vNH_loc:")
print(res["vNH_loc"])
print("vNH_trans:")
print(res["vNH_trans"])
print("vH_loc:")
print(res["vH_loc"])
print("vH_trans sum:", res["vH_trans"].sum())
print("Residuals:")
print(res["residuals"]["max_abs"])

total_mass = (
    res["vNH_loc"].sum()
    + res["vNH_trans"].sum()
    + res["vH_loc"].sum()
    + res["vH_trans"].sum()
)
print("Total mass:", total_mass)
print("Target mass n:", n)

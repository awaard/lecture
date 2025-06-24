import numpy as np

def pinv_svd(J, *,
             variant: str = "plain",
             eps: float = 1e-4,
             lam: float = 0.03,
             lam_min: float = 0.01):
    
    U, S, Vt = np.linalg.svd(J, full_matrices=False)

    if variant == "plain":
        Sinv = [1/s if s > 0 else 0 for s in S]

    elif variant == "clipped":
        # Moore–Penrose but *zero* out small σ
        Sinv = [0 if s < eps else 1/s for s in S]

    elif variant == "damped":
        # J⁺ = Jᵀ · (J Jᵀ + λ²I)⁻¹  ⇒  σᵢ /(σᵢ² + λ²)
        Sinv = [s/(s**2 + lam**2) for s in S]

    elif variant == "smooth":
        # Use MP for large σ, damped formula for small σ (continuous switch)
        Sinv = [1/s if s > eps else s/(s**2 + lam_min**2) for s in S]

    else:
        raise ValueError(f"Unknown variant '{variant}'")

    return Vt.T @ np.diag(Sinv) @ U.T
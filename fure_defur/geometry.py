"""Shell calibration and bounded normal-displacement defurring.

All geometry is in scene units. Correspondence weights are heuristic support
scores, not calibrated probabilities or measurements of physical fur depth.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.spatial import cKDTree

ZERO_PARTS = {"eyes", "nosetip", "nose", "paw_pads", "horns", "inner_earcanal", "unlabeled"}
MULTIPLIERS = {"belly": 3.6, "body": 3.4, "neck": 3.5, "mane": 3.8,
               "face": 3.0, "ears": 3.0, "leg_front": 2.6, "leg_rear": 2.6,
               "front_paws": 2.2, "paws": 2.15, "tail": 2.75, "under_tail": 2.7}


def calibrate_lengths(widths: np.ndarray, labels: np.ndarray, names: list[str],
                      cm_per_unit: float, vlm: Mapping[str, float]) -> dict:
    """Aggregate nearest-shell widths and return both length initializations."""
    widths = np.asarray(widths, dtype=np.float64)
    if widths.shape != labels.shape:
        raise ValueError("widths and labels must have the same shape")
    if not np.isfinite(widths).all():
        raise ValueError("widths must contain only finite values")
    if np.any(widths < 0):
        raise ValueError("widths must be nonnegative")
    if not np.isfinite(cm_per_unit) or cm_per_unit <= 0:
        raise ValueError("cm_per_unit must be finite and positive")
    out = {}
    for p, name in enumerate(names):
        mask = labels == p
        if not np.any(mask):
            continue
        raw = float(np.percentile(widths[mask], 75) * cm_per_unit)
        thickness = 0.0 if name in ZERO_PARTS else raw
        multiplier = 0.0 if name in ZERO_PARTS else MULTIPLIERS.get(name, 2.75)
        v1 = thickness * multiplier
        reference = float(vlm.get(name, 0.0))
        if not np.isfinite(reference) or reference < 0:
            raise ValueError(f"Reference length for {name!r} must be finite and nonnegative")
        v2 = 0.0 if v1 == 0 else (float(np.clip(.6*v1 + .4*reference, .35*reference, 1.35*reference))
                                  if reference > 0 else v1)
        out[name] = dict(vertex_count=int(mask.sum()), raw_shell_p75_cm=raw,
                         effective_thickness_cm=thickness, multiplier=multiplier,
                         v1_length_cm=v1, v2_length_cm=v2, reference_length_cm=reference,
                         v1_width_cm=0.0 if v1 == 0 else float(np.clip(.03*v1, .02, .12)),
                         v2_width_cm=0.0 if v2 == 0 else float(np.clip(.03*v2, .02, .12)))
    return out


def validate_shell(shell: Mapping[str, np.ndarray]) -> dict:
    """Validate stored widths, signed offsets, and unit normal magnitudes."""
    v, n = np.asarray(shell["verts"]), np.asarray(shell["normals"])
    width = np.asarray(shell["shell_width"]).reshape(-1)
    inner, outer = [np.asarray(shell[k]).reshape(-1) for k in ("inner_dist", "outer_dist")]
    if v.shape != n.shape or v.shape != (len(width), 3):
        raise ValueError("Shell vertices, normals, and widths have incompatible shapes")
    if inner.shape != width.shape or outer.shape != width.shape:
        raise ValueError("Shell offsets and widths have incompatible shapes")
    if not all(np.isfinite(x).all() for x in (v, n, width, inner, outer)):
        raise ValueError("Shell arrays must contain only finite values")
    if np.any(width < 0):
        raise ValueError("Shell widths must be nonnegative")
    if not np.allclose(width, np.abs(outer-inner), atol=1e-6, rtol=1e-4):
        raise ValueError("Shell widths do not match the stored inner and outer offsets")
    normal_norm = np.linalg.norm(n, axis=1)
    if not np.allclose(normal_norm, 1, atol=1e-3):
        raise ValueError("Shell normals must have unit length")
    return dict(vertex_count=len(v), width_p50=float(np.median(width)),
                width_p95=float(np.percentile(width, 95)), width_max=float(width.max()),
                finite_fraction=1.0, normal_norm_max_error=float(np.max(abs(normal_norm-1))))


def normals_and_edges(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute area-weighted vertex normals without changing vertex/face ordering."""
    tri = vertices[faces]
    fn = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float64)
    for k in range(3):
        np.add.at(normals, faces[:, k], fn)
    norm = np.linalg.norm(normals, axis=1)
    if np.any(norm <= 1e-14):
        raise ValueError("Source mesh has a degenerate vertex normal; repair it before defurring")
    normals /= norm[:, None]
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    return normals, np.unique(edges, axis=0)


def transfer_widths(vertices: np.ndarray, normals: np.ndarray, labels: np.ndarray,
                    shell: Mapping[str, np.ndarray], radius: float, k: int = 12) -> dict:
    """Transfer shell widths through part, normal, distance, and outlier gates."""
    sv = np.asarray(shell["verts"], dtype=np.float64)
    sn = np.asarray(shell["normals"], dtype=np.float64)
    sw = np.asarray(shell["shell_width"]).reshape(-1)
    reliable = np.asarray(shell["reliability_mask_init"]).reshape(-1).astype(bool)
    # Shell labels are derived from this same annotated mesh; they are not new ground truth.
    _, shell_to_mesh = cKDTree(vertices).query(sv, workers=4)
    shell_labels = labels[shell_to_mesh]
    distances, indices = cKDTree(sv).query(vertices, k=min(k, len(sv)), workers=4)
    if distances.ndim == 1:
        distances, indices = distances[:, None], indices[:, None]
    alignment = np.einsum("ij,ikj->ik", normals, sn[indices])
    valid = ((shell_labels[indices] == labels[:, None]) & reliable[indices]
             & (alignment > .3) & (distances < radius))
    weights = valid * np.exp(-np.square(distances/radius)) * np.square(np.clip((alignment-.3)/.7, 0, 1))
    den = weights.sum(axis=1)
    transferred = np.divide((weights*sw[indices]).sum(axis=1), den,
                            out=np.zeros(len(vertices)), where=den > 1e-12)
    return dict(width=transferred, confidence=weights.max(axis=1),
                nearest_width=sw[indices[:, 0]], nearest_distance=distances[:, 0],
                nearest_reliable=reliable[indices[:, 0]],
                nearest_alignment=alignment[:, 0], supported=den > 1e-12)


def align_shell_normal_frame(vertices: np.ndarray, normals: np.ndarray,
                             shell: Mapping[str, np.ndarray], radius: float) -> tuple[dict, dict]:
    """Resolve one global shell-normal sign for correspondence.

    Only normals used for correspondence are changed. Stored offsets and widths
    remain untouched; this copy must not be used to reconstruct shell surfaces.
    """
    distance, ids = cKDTree(vertices).query(shell["verts"], workers=4)
    alignment = np.einsum("ij,ij->i", normals[ids], shell["normals"])
    valid = (distance < radius) & np.asarray(shell["reliability_mask_init"]).reshape(-1).astype(bool)
    if valid.sum() < min(20, len(vertices)):
        raise RuntimeError("Insufficient nearby support to resolve shell normal frame")
    median = float(np.median(alignment[valid]))
    if abs(median) <= .3:
        raise RuntimeError("Ambiguous shell winding: inspect alignment before transfer")
    sign = 1 if median > 0 else -1
    copy = dict(shell)
    copy["normals"] = np.asarray(shell["normals"])*sign
    return copy, dict(normal_sign=sign, original_median_dot=median,
                      nearby_reliable_samples=int(valid.sum()), correspondence_only=True)


def part_calibrated_targets(width: np.ndarray, labels: np.ndarray, names: list[str],
                            priors_scene: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend pooled shell width and the configured part-thickness value."""
    clipped, target, ratio = width.copy(), np.zeros_like(width), np.zeros_like(width)
    for p, name in enumerate(names):
        mask = labels == p
        if not mask.any():
            continue
        clipped[mask] = np.minimum(clipped[mask], np.percentile(clipped[mask], 95))
        p75, p95 = np.percentile(clipped[mask], [75, 95])
        prior = float(priors_scene.get(name, p75))
        t = 0.0 if name in ZERO_PARTS else float(np.clip(.65*p75 + .35*prior, 0, p95))
        target[mask] = t
        ratio[mask] = 0.0 if p75 <= 1e-8 else min(t/p75, 1.15)
    return clipped, target, ratio


def laplacian(vertices: np.ndarray, edges: np.ndarray, labels: np.ndarray) -> sparse.csr_matrix:
    """Build a weighted graph Laplacian with weaker coupling across part boundaries."""
    a, b = edges.T
    length = np.linalg.norm(vertices[a]-vertices[b], axis=1)
    weight = np.clip(np.median(length)/np.maximum(length, 1e-12), .1, 10)
    weight *= np.where(labels[a] == labels[b], 1.0, .15)
    adjacency = sparse.coo_matrix((np.r_[weight, weight], (np.r_[a, b], np.r_[b, a])),
                                 shape=(len(vertices), len(vertices))).tocsr()
    return sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency


def solve_displacement(target: np.ndarray, prior: np.ndarray, confidence: np.ndarray,
                       upper: np.ndarray, pinned: np.ndarray, graph: sparse.csr_matrix,
                       smoothness: float = 8.0) -> tuple[np.ndarray, dict]:
    """Solve a convex bounded quadratic in displacement, not mesh coordinates."""
    if not all(x.shape == target.shape for x in (prior, confidence, upper, pinned)):
        raise ValueError("Displacement inputs must have the same shape")
    if not all(np.isfinite(x).all() for x in (target, prior, confidence, upper)):
        raise ValueError("Displacement inputs must contain only finite values")
    if np.any(upper < 0):
        raise ValueError("Displacement upper bounds must be nonnegative")
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("Displacement confidence must be between zero and one")
    if not np.isfinite(smoothness) or smoothness < 0:
        raise ValueError("smoothness must be finite and nonnegative")
    scale = max(float(np.max(upper)), 1e-8)
    bounds = upper/scale
    bounds = bounds.copy()
    bounds[pinned] = 0
    weight = confidence + .25
    rhs = (confidence*target + .25*prior)/scale
    matrix = sparse.diags(weight) + smoothness*graph

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        product = matrix @ x
        return float(.5*x.dot(product) - rhs.dot(x)), product-rhs

    result = minimize(objective, np.clip(prior/scale, 0, bounds), method="L-BFGS-B", jac=True,
                      bounds=np.column_stack([np.zeros(len(bounds)), bounds]),
                      options={"maxiter": 2500, "ftol": 1e-13, "gtol": 1e-7, "maxcor": 8})
    gradient = objective(result.x)[1]
    projected = result.x - np.clip(result.x-gradient, 0, bounds)
    residual = float(np.max(np.abs(projected)))
    if not np.isfinite(result.x).all() or not np.isfinite(residual) or residual >= 2e-3:
        raise RuntimeError(
            f"Displacement solve did not converge: {result.message}; "
            f"projected gradient infinity norm={residual}"
        )
    displacement = result.x*scale
    displacement[pinned] = 0
    return displacement, dict(iterations=int(result.nit), success=bool(result.success),
                              projected_gradient_inf=residual, smoothness=smoothness,
                              prior_weight=.25, objective=float(result.fun))


def legacy_smooth(raw: np.ndarray, faces: np.ndarray, labels: np.ndarray,
                  pinned: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Apply 35 same-part and 5 global scalar-smoothing steps."""
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges = np.unique(edges, axis=0)
    values = raw.copy()
    for same_part, count, alpha in ((True, 35, .45), (False, 5, .12)):
        e = edges[labels[edges[:, 0]] == labels[edges[:, 1]]] if same_part else edges
        a, b = e.T
        adj = sparse.coo_matrix((np.ones(2*len(e)), (np.r_[a, b], np.r_[b, a])),
                               shape=(len(raw), len(raw))).tocsr()
        degree = np.asarray(adj.sum(axis=1)).ravel()
        active = degree > 0
        for _ in range(count):
            mean = np.divide(adj @ values, degree, out=values.copy(), where=active)
            values = (1-alpha)*values + alpha*mean
    values[pinned] = 0
    return np.clip(values, 0, upper)


def safe_displacement(vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray,
                       displacement: np.ndarray) -> tuple[np.ndarray, dict]:
    """Local backtracking rejects relative face flips and severe area collapse.

    This is not a self-intersection proof; intersections are checked separately.
    """
    tri = vertices[faces]
    original = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    area2 = np.linalg.norm(original, axis=1)
    if np.any(area2 <= 1e-16):
        raise ValueError("Source has degenerate faces")
    d = displacement.copy()
    affected = np.zeros(len(d), dtype=bool)
    for iteration in range(80):
        t = (vertices-d[:, None]*normals)[faces]
        current = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
        bad = ((np.einsum("ij,ij->i", current, original) <= .05*area2**2)
               | (np.linalg.norm(current, axis=1) < .1*area2))
        if not bad.any():
            return d, dict(backtrack_iterations=iteration, affected_vertex_fraction=float(affected.mean()))
        ids = np.unique(faces[bad])
        d[ids] *= .5
        affected[ids] = True
    raise RuntimeError("Could not obtain a non-inverting normal displacement")

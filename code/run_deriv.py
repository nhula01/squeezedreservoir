"""Derivative convergence check: fixed-step central difference (projected at boundaries, as used in the early studies) vs the
convergent Richardson/one-sided estimator, against a fine reference, at interior and boundary anchors."""
from common import *
anchors = {'interior_mid': np.array([0.25, 0.25, 1.0]), 'interior_basin': np.array([0.05, 0.2, -0.52]), 'boundary_r0': np.array([0.0, 0.2, -0.52]), 'boundary_re0': np.array([0.15, 0.0, 0.0])}
rows = []
for name, th in anchors.items():
    cnt = Counter(DEV, MG); g_fixed = fd_gradient(cnt, th)                                   # old estimator (projection at the boundary)
    cnt2 = Counter(DEV, MG); g_conv, err, hs = fd_gradient_conv(cnt2, th, return_info=True)   # convergent estimator
    cnt3 = Counter(DEV, MG); g_ref, err_ref, _ = fd_gradient_conv(cnt3, th, h0=np.array([2e-3, 2e-3, 2e-3]), tol=1e-3, max_halvings=2, return_info=True)  # fine reference
    rows.append([name, th, g_fixed, g_conv, err, hs, g_ref, err_ref, cnt.n, cnt2.n])
    log(name, 'fixed', g_fixed.round(6), 'conv', g_conv.round(6), '+-', err.round(7), 'ref', g_ref.round(6), 'evals', cnt.n, cnt2.n)
np.savez(os.path.join(DATA, 'deriv.npz'), rows=np.array(rows, dtype=object))

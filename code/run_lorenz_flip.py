"""Lorenz-63 with the opposite parametric phase (phi_s = pi, s = -1): the input strength becomes eps e^{+r}."""
from common import *
from run_tasks import coarse_scan, R3, RE3, D3
from sqz import make_lorenz, effective_parameters
task = make_lorenz(horizon=2); dev = replace(DEV, s=-1.0)
L3 = np.zeros((6, 6, 12)); N3 = L3.copy(); M3 = L3.copy()
for i, r in enumerate(R3):
    for j, re in enumerate(RE3):
        for k, d in enumerate(D3):
            L3[i, j, k], N3[i, j, k], M3[i, j, k] = evaluate(dev, (r, re, d), task)
    log('lorenz flip scan', i, L3[i].min())
i, j, k = np.unravel_index(L3.argmin(), L3.shape); best = np.array([R3[i], RE3[j], D3[k]])
e1 = effective_parameters(dev, best)
conv = [evaluate(replace(dev, Nc=Nc), best, task) for Nc in (8, 12, 16)]
np.savez(os.path.join(DATA, 'lorenz_flip.npz'), L3=L3, N3=N3, M3=M3, best=best, eff1=np.array([[k_, v] for k_, v in e1.items()], dtype=object), conv=np.array(conv))
log('done', best, L3.min(), conv)

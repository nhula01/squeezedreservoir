"""Best base device first, then squeezing: for each task, (1) optimize the free drive strength eps at zero squeezing
(the drive amplitude and phase are set by the experimenter), (2) search the squeezing controls (r, r_e, dth) and the drive
phase phi_d in {0, pi/2} from that base, (3) refine with convergent-gradient descent, (4) Fock check at the optimum."""
from common import *
from sqz import make_lorenz, make_nce, Task
R3, RE3, D3 = np.linspace(0, 0.5, 4), np.linspace(0, 0.5, 4), np.linspace(-np.pi, np.pi, 9)[:-1]
import reservoirpy.datasets as ds
X = ds.santafe_laser().ravel().astype(float)[:1000]
TASKS = {'mg': MG, 'narma': NARMA, 'lorenz': make_lorenz(horizon=2), 'nce': make_nce(), 'laser': Task('laser', X, np.roll(X, -1), 100, 700, 1)}
EPS = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
NMAX = 2.5      # photon budget: operating points with max <a^dag a> above this are excluded (truncation reliability / pump-power budget)
class Budgeted:
    """Counter that returns a large loss for points violating the photon budget (so the optimizer never enters them)."""
    def __init__(self, dev, task, budget): self.cnt = Counter(dev, task, budget=budget)
    def __call__(self, th):
        L = self.cnt(th); nm = self.cnt.trace[-1][6]
        return L if nm <= NMAX else L * (1 + 10 * (nm - NMAX))

def nc_for(eps, squeezed=False): return (8 if eps <= 0.3 else 10 if eps <= 0.5 else 12) if not squeezed else 12

if __name__ == '__main__':
    which = sys.argv[1:] or list(TASKS)
    fp = os.path.join(DATA, 'base.npz'); out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
    for name in which:
        task = TASKS[name]
        if f'{name}_eps' not in out:                                    # (1) base optimization of the drive strength
            rows = np.array([[e, *evaluate(replace(DEV, eps=e, Nc=nc_for(e)), (0, 0, 0), task)] for e in EPS])
            e0 = EPS[rows[:, 1].argmin()]; fine = np.unique(np.clip([e0 * f for f in (0.8, 0.9, 1.1, 1.25)], 0.05, 1.0))
            rows2 = np.array([[e, *evaluate(replace(DEV, eps=e, Nc=nc_for(e)), (0, 0, 0), task)] for e in fine])
            allr = np.vstack([rows, rows2]); ok = allr[:, 3] <= NMAX; eb = allr[ok][allr[ok][:, 1].argmin(), 0]
            out[f'{name}_eps'] = allr; out[f'{name}_epsbest'] = eb; log(name, 'base eps search (photon budget %.1f): best eps %.3f, L %.4e (eps=0.2: %.4e), nmax %.2f' % (NMAX, eb, allr[ok][:, 1].min(), rows[EPS == 0.2][0, 1], allr[allr[:, 0] == eb][0, 3])); np.savez(fp, **out)
        eb = float(out[f'{name}_epsbest']); dev = replace(DEV, eps=eb, Nc=nc_for(eb, True))
        if f'{name}_L3' not in out:                                     # (2a) coarse scan, phi_d = 0
            L3 = np.zeros((4, 4, 8)); N3 = L3.copy(); M3 = L3.copy()
            for i, r in enumerate(R3):
                for j, re in enumerate(RE3):
                    for k, d in enumerate(D3): L3[i, j, k], N3[i, j, k], M3[i, j, k] = evaluate(dev, (r, re, d), task)
                log(name, 'scan phi_d=0', i, L3[i].min())
            L3 = np.where(M3 <= NMAX, L3, np.inf); out[f'{name}_L3'], out[f'{name}_N3'], out[f'{name}_M3'] = L3, N3, M3; np.savez(fp, **out)
        if f'{name}_G2' not in out:                                     # (2b) (r, re) grids at phi_d = pi/2 for three phases
            L3 = out[f'{name}_L3']; i, j, k = np.unravel_index(L3.argmin(), L3.shape); phases = [D3[k], 0.0, np.pi]
            G = np.zeros((3, 4, 4, 3))
            for p, d in enumerate(phases):
                for i, r in enumerate(R3):
                    for j, re in enumerate(RE3): G[p, i, j] = evaluate(replace(dev, phi_d=np.pi / 2), (r, re, d), task)
                log(name, 'grid phi_d=pi/2 dth=%.2f' % d, G[p, :, :, 0].min())
            G[:, :, :, 0] = np.where(G[:, :, :, 2] <= NMAX, G[:, :, :, 0], np.inf); out[f'{name}_G2'], out[f'{name}_G2phases'] = G, np.array(phases); np.savez(fp, **out)
        if f'{name}_gd' not in out:                                     # (3) convergent-gradient descent from the two best grid points
            L3, G = out[f'{name}_L3'], out[f'{name}_G2']; cands = []
            i, j, k = np.unravel_index(L3.argmin(), L3.shape); cands.append((L3.min(), 0.0, np.array([R3[i], RE3[j], D3[k]])))
            p, i, j = np.unravel_index(G[:, :, :, 0].argmin(), G.shape[:3]); cands.append((G[p, i, j, 0], np.pi / 2, np.array([R3[i], RE3[j], out[f'{name}_G2phases'][p]])))
            res = []
            for L_, pd, th0 in cands:
                bf = Budgeted(replace(dev, phi_d=pd), task, 80); gradient_descent_conv(bf, th0, 80); tr = np.array(bf.cnt.trace); tr = tr[tr[:, 6] <= NMAX]; ib = tr[:, 4].argmin()
                res.append([pd, *tr[ib, 1:7], L_]); log(name, 'GD phi_d=%.2f from' % pd, th0.round(2), 'L %.4e -> %.4e at' % (L_, tr[ib, 4]), tr[ib, 1:4].round(3))
            out[f'{name}_gd'] = np.array(res); np.savez(fp, **out)
        if f'{name}_best' not in out:                                   # (4) best point + Fock check
            res = out[f'{name}_gd']; b = res[res[:, 4].argmin()]; pd, th = b[0], b[1:4]
            conv = [evaluate(replace(dev, phi_d=pd, Nc=Nc), th, task) for Nc in (dev.Nc, dev.Nc + 4)]
            out[f'{name}_best'], out[f'{name}_bestpd'], out[f'{name}_conv'] = th, pd, np.array(conv)
            L0 = out[f'{name}_eps'][:, 1].min(); log(name, 'RESULT base L %.4e -> squeezed %.4e (%.1f%%), phi_d %.2f, theta' % (L0, conv[0][0], 100 * (1 - conv[0][0] / L0), pd), th.round(3), 'Nc check', conv); np.savez(fp, **out)
    log('done base', which)

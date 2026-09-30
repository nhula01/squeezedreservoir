"""Build the documentation notebooks (one per study) from the project code and saved data."""
import nbformat as nbf, os
NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'notebooks'); os.makedirs(NB, exist_ok=True)

HEADER = '''import os, sys, subprocess, numpy as np
CODE = os.path.abspath(os.path.join(os.getcwd(), '..', 'code')); sys.path.insert(0, CODE); os.chdir(CODE); os.environ['OMP_NUM_THREADS'] = '1'
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
from common import *
import make_figs
RUN = False   # set True to recompute the data for this notebook (see the "Compute" section for the cost)
def show(pdf, dpi=110):
    """Render a figure PDF from paper/figs inline."""
    base = '/tmp/' + os.path.basename(pdf)[:-4]; subprocess.run(['pdftoppm', '-r', str(dpi), '-png', '-singlefile', pdf, base], check=True); display(Image(filename=base + '.png'))
def ld(name):
    p = os.path.join(DATA, name); return np.load(p, allow_pickle=True) if os.path.exists(p) else None
'''

def nb(title, cells, fname):
    n = nbf.v4.new_notebook(); n.cells = [nbf.v4.new_markdown_cell(f'# {title}')]
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == 'md' else nbf.v4.new_code_cell(src))
    nbf.write(n, os.path.join(NB, fname))

# ---------------------------------------------------------------------------------------------- 00 model
nb('Squeezed cavity-QED reservoir: model, simulator and effective device parameters', [
('md', r'''## Statement of the problem
A physical reservoir computer maps an input time series into the transient response of a dynamical system and trains only a linear readout. Its quality depends on device parameters (input strength, coupling, detunings, relaxation of the nonlinear element) that are fixed at fabrication. **Can optical squeezing turn these fabrication parameters into post-fabrication controls?**

The device is a single cavity mode $a$ coupled to a two-level emitter $\sigma$, parametrically driven and coupled to a squeezed bath. In the frame rotating at half the pump frequency ($\phi_s=0$):
$$H_0=\omega_a a^\dagger a+\omega_q\sigma^\dagger\sigma-\tfrac{\lambda}{2}(a^2+a^{\dagger2})+g(a^\dagger\sigma+a\sigma^\dagger),\qquad H_{\rm in}=i\epsilon f(t)(a-a^\dagger),$$
$$\dot\rho=-i[H_0+H_{\rm in},\rho]+\mathcal D[L_a]\rho+\gamma\mathcal D[\sigma]\rho,\qquad L_a=\sqrt\kappa(\cosh r_e\,a+e^{i\theta}\sinh r_e\,a^\dagger).$$
Controls: $\vartheta=(r, r_e, \Delta\theta)$ with $\tanh 2r=\lambda/\omega_a$, $\Delta\theta=\theta-\phi_s$. Fabricated and fixed: $g,\kappa,\gamma,\epsilon,\omega_a,\omega_q$.

## Simulation method
* **Bogoliubov frame.** $a=\alpha\cosh r+\alpha^\dagger\sinh r$ diagonalizes the quadratic cavity part exactly: $\Omega_a=\omega_a/\cosh 2r$; the drive becomes $i\epsilon e^{-r}f(\alpha-\alpha^\dagger)$; the coupling $g\cosh r(\alpha^\dagger\sigma+{\rm h.c.})+g\sinh r(\alpha\sigma+{\rm h.c.})$. Truncation ($N_c=8$) is applied to the quasi-mode $\alpha$.
* **Real Hermitian basis.** Superoperators act on the real vector space of Hermitian matrices (dimension $D^2=324$), halving memory and quartering flops versus the complex vectorization.
* **Chebyshev-interpolated propagator.** The Liouvillian is linear in the input, $\mathcal L(f)=\mathcal L_0+f\mathcal L_1$; $e^{\mathcal L(f)\tau}$ is evaluated exactly (`scipy.linalg.expm`) at 7 Chebyshev nodes $f_k\in[-1,1]$ and reconstructed at each input by barycentric interpolation (error $<10^{-10}$).
* **Readout.** Six homodyne-accessible observables $Q,P,Q^2,P^2,\sigma_x,\sigma_y$ at $N_v=4$ virtual times per symbol, standardized, ridge with $\delta=N_{\rm tr}\sigma_{\rm rel}^2$, $\sigma_{\rm rel}=1\%$ (noise-matched).

The whole simulator is `sqz.py`; the device and task constants are in `common.py`.'''),
('code', HEADER),
('code', '''from sqz import *
print(DEV)
task = MG
t0 = time.time(); L, nr, nmax = evaluate(DEV, (0, 0, 0), task); print('unsqueezed device on Mackey-Glass: L = %.4e, held-out NRMSE = %.3f, max photons = %.3f, time %.2f s' % (L, nr, nmax, time.time() - t0))'''),
('md', r'''## Accuracy checks of the propagation
Chebyshev order and Fock cutoff (a corner of the control box is the most demanding point, because for $\Delta\theta=0$ the two squeezings add).'''),
('code', '''import sqz
res = Reservoir(replace(DEV, K=9), (0.5, 0.5, 0.0)); rng = np.random.default_rng(1); rho = res.rho0.copy()
for u in rng.uniform(-1, 1, 30): rho = res.exact_step(rho, u)
for K in (5, 7, 9):
    rk = Reservoir(replace(DEV, K=K), (0.5, 0.5, 0.0)); errs = []
    for u in rng.uniform(-1, 1, 5):
        C = sqz._cheb_coeffs(np.array([u]), rk.x, rk.w); P = (C[0] @ rk.Pflat).reshape(rk.D**2, rk.D**2); errs.append(np.abs(P @ rho - rk.exact_step(rho, u)).max())
    print('K = %d: max interpolation error %.1e' % (K, max(errs)))
for th in ((0, 0, 0), (0.02, 0.2, -0.52), (0.5, 0.5, 0.0)):
    print('theta', th, ['Nc=%d: L=%.5e n=%.3f' % (Nc, *[evaluate(replace(DEV, Nc=Nc), th, MG)[i] for i in (0, 2)]) for Nc in (8, 12)])'''),
('md', r'''## What each control does to the effective device
* $r$: input strength $\epsilon e^{\mp r}$ (sign set by the parametric phase $\phi_s$), couplings $g\cosh r$, $g\sinh r$, cavity frequency $\omega_a/\cosh 2r$.
* $r_e$, $\Delta\theta$: **mean-field lemma** $\frac{d}{dt}\langle a\rangle|_{\rm bath}=-\frac{\kappa}{2}\langle a\rangle$ for any $r_e,\theta$ (because $\cosh^2 r_e-\sinh^2 r_e=1$) — the external squeezing does not touch the cavity decay or the linear response; it acts only through the emitter, whose relaxation rate it sets, while the phase orients the intracavity noise.

`effective_parameters` computes $\epsilon_{\rm eff},\Omega_a,g\cosh r,g\sinh r$, the integrated emitter relaxation rates $\gamma^{\rm eff}_{x,y,z}$ and stationary variances; `run_review.liouvillian_gap` the spectral gap of $\mathcal L_0$ (equal to $(\kappa+\gamma)/4$ at zero squeezing).'''),
('code', '''from sqz import effective_parameters
from run_review import liouvillian_gap
for th in ((0, 0, 0), (0, 0.2, -0.79), (0, 0.5, -0.79), (0.3, 0, 0)):
    e = effective_parameters(DEV, th); print(th, {k: round(float(v), 3) for k, v in e.items()}, 'gap %.3f' % liouvillian_gap(DEV, th))'''),
('md', '## Compute\n`run_tasks.py` (effective-parameter lines, ~1 min) and `run_review.py gap` (~1 min) produce `data/tasks.npz` and `data/review.npz`; with `RUN=False` the saved data are used.'),
('code', '''if RUN:
    subprocess.run(['python3', 'run_tasks.py']); subprocess.run(['python3', 'run_review.py', 'gap'])
make_figs.fig1(); make_figs.fig1_effective(); show('paper/figs/fig1_sketch.pdf'); show('paper/figs/fig1_effective.pdf')'''),
('md', '**Figure 1c.** Linear device parameters against $r$ (left); emitter relaxation rates, Liouvillian gap and stationary quadrature variances against $r_e$ (middle) and against $\\Delta\\theta$ (right). The rates move with $r_e$, are phase independent, and the phase exchanges Var$(Q)$ and Var$(P)$.'),
], '00_model_and_effective_parameters.ipynb')

# ---------------------------------------------------------------------------------------------- 01 landscape
nb('Task-loss landscape over the squeezing controls (Mackey–Glass)', [
('md', r'''## Statement of the problem
Does squeezing merely rescale the reservoir output, or does it reshape the task loss $\mathcal L(\vartheta)=\frac{1}{2N_{\rm tr}}\|y-X(\vartheta)w^\star(\vartheta)\|^2$ (readout refit at every setting) in a way that experimentally accessible controls can navigate?

**Task.** Ten-step Mackey–Glass prediction ($\beta,\gamma,n,\tau)=(0.2,0.1,10,17)$, unit-step Euler, 1000 samples: 100 washout, 700 training, 190 held out.

**Protocol.** Coarse $6\times6\times12$ scan of $(r,r_e,\Delta\theta)\in[0,0.5]^2\times[-\pi,\pi]$; dense $25\times25$ slices $\mathcal L(r,\Delta\theta)$ at the best $r_e$ and $\mathcal L(r,r_e)$ at the best phase; projected finite-difference gradient descent restricted to each slice (4 random starts, 60 evaluations); Fock-cutoff mask (cells where $\mathcal L$ changes $>1\%$ between $N_c=8$ and 12).'''),
('code', HEADER),
('md', '## Compute\n`run_fig2.py` (~25 min on one core) writes `data/fig2_scans.npz` and `data/fig2_paths.npz`.'),
('code', '''if RUN: subprocess.run(['python3', 'run_fig2.py'])
sc = ld('fig2_scans.npz'); L3 = sc['L3']; L0 = L3[0, 0, 0]; i, j, k = np.unravel_index(L3.argmin(), L3.shape)
print('unsqueezed L0 = %.4e (NRMSE %.3f)' % (L0, sc['N3'][0, 0, 0]))
print('coarse optimum (r, re, dth) = (%.2f, %.2f, %.2f): L = %.4e, reduction %.1f%%' % (sc['r3'][i], sc['re3'][j], sc['d3'][k], L3.min(), 100 * (1 - L3.min() / L0)))
print('dense-slice optimum: L = %.4e (%.1f%%), NRMSE %.3f' % (sc['L1'].min(), 100 * (1 - sc['L1'].min() / L0), sc['NR1'].ravel()[sc['L1'].argmin()]))
print('intracavity squeezing alone (re=0): best %.1f%% at r = %.2f' % (100 * (1 - sc['L2'][:, 0].min() / L0), sc['rg'][sc['L2'][:, 0].argmin()]))'''),
('code', "make_figs.fig2(); show('paper/figs/fig2_landscape.pdf')"),
('md', r'''**Figure 2.** (a) $\mathcal L(r,\Delta\theta)$ at $r_e=0.2$ with gradient trajectories (squares: start, stars: end). (b) $\mathcal L(r,r_e)$ at $\Delta\theta=-0.79$; $\times$ unsqueezed; hatched cells fail the Fock-cutoff check. (c) Held-out NRMSE shares the basin. (d) Along $r_e=0$, intracavity squeezing alone has an interior optimum near $r\approx0.14$; with external squeezing on, the optimum moves to $r\approx0$.

Interpretation: the $(r_e,\Delta\theta)$ basin is the emitter channel (phase-sensitive relaxation), the $r$ dependence is the linear-response channel (input de-amplified by $e^{-r}$, cavity detuned).'''),
], '01_landscape_mackey_glass.ipynb')

# ---------------------------------------------------------------------------------------------- 02 optimizers
nb('Physical gradient training: optimizer comparison at matched evaluation budget', [
('md', r'''## Statement of the problem
Gradients with respect to the squeezing controls are obtained by repeating the reservoir experiment at nearby settings, $\partial\mathcal L/\partial\vartheta_i\approx[\mathcal L(\vartheta+h e_i)-\mathcal L(\vartheta-h e_i)]/2h$, followed by projected gradient descent with backtracking. Is this more efficient than derivative-free search when **every** reservoir evaluation (probes and rejected steps included) is counted?

**Protocol.** 12 random initial settings, 100 reservoir evaluations each, three methods: finite-difference gradient descent ($h=10^{-2}$), Nelder–Mead (bounded), uniform random search. Best-so-far training loss recorded after every evaluation.

This comparison is not the point of the paper (it lives in Supplementary Note 7): in three controls the 5 %-basin is broad and any method finds it; the gradient only refines better.'''),
('code', HEADER),
('md', '## Compute\n`run_fig3.py mg` (~35 min).'),
('code', '''if RUN: subprocess.run(['python3', 'run_fig3.py', 'mg'])
f3 = ld('fig3_mg.npz'); sc = ld('fig2_scans.npz'); L0 = sc['L3'][0, 0, 0]
Lstar = min(sc['L3'].min(), sc['L1'].min(), sc['L2'].min(), *[f3[f'final_{k}'][:, 3].min() for k in ('gd', 'nm', 'rs')])
for k, name in (('gd', 'gradient descent'), ('nm', 'Nelder-Mead'), ('rs', 'random search')):
    C = f3[f'curve_{k}']; reach5 = [(np.argmax(c <= 1.05 * Lstar) + 1) if np.any(c <= 1.05 * Lstar) else np.inf for c in C]
    print('%-16s median reduction %.0f%%; within 5%% of L* after median %s evals; within 1%% in %d/%d runs; median final %.3e' % (name, 100 * np.median(1 - C[:, -1] / C[:, 0]), np.median(reach5), np.sum(C[:, -1] <= 1.01 * Lstar), len(C), np.median(C[:, -1])))'''),
('code', "make_figs.fig3('mg', 'fig3_optim.pdf', L0, Lstar); show('paper/figs/fig3_optim.pdf')"),
('md', '**Supplementary Figure 5.** (a) Median best-so-far loss; (b) final-loss distributions; (c) accepted gradient-descent paths in the $(r_e,\\Delta\\theta)$ plane. Two runs of each local method are trapped at boundary minima of the $r$ channel; two converge to a secondary basin near $\\Delta\\theta\\approx\\pi$.'),
], '02_optimizers.ipynb')

# ---------------------------------------------------------------------------------------------- 03 mechanism
nb('Which observables see which control: feature sensitivity, QFI and task gradient', [
('md', r'''## Statement of the problem
The mean-field lemma predicts that external squeezing is invisible in $\langle Q\rangle,\langle P\rangle$ and reaches the readout only through the second moments and the emitter. Test this on the features, and check the relation between quantum Fisher information (how much the state moves) and the task gradient (how much the loss moves), which the companion Fisher-bound paper says are related by conversion factors, not equal.

**Quantities** along lines through the Mackey–Glass basin: per-sample feature response $S_i=N_{\rm tr}^{-1/2}\|\partial X/\partial\vartheta_i\|_F$ (standardized, bias-centred), task-gradient magnitude $G_i=|\partial\mathcal L/\partial\vartheta_i|$, diagonal QFI $F_{ii}(t)=2\sum_{pq}|\langle p|\partial_i\rho_t|q\rangle|^2/(\lambda_p+\lambda_q)$ averaged over the training window; decomposition of $S_i^2$ by observable class; full gradient descent initialized along the lines.'''),
('code', HEADER),
('md', '## Compute\n`run_fig4.py` (~15 min).'),
('code', '''if RUN: subprocess.run(['python3', 'run_fig4.py'])
d = ld('fig4_lines.npz'); Sg = d['line_r'][0, 12:21].reshape(3, 3); fr = Sg**2 / (Sg**2).sum(1, keepdims=True)
for i, c in enumerate(('r', 're', 'dth')): print('S_%s^2 shares: means %.1f%%, second moments %.1f%%, emitter %.1f%%' % (c, *(100 * fr[i])))
print('median QFI per sample along re: %.2f; along dth: %.1e' % (np.median(d['line_re'][:, 7]), np.median(d['line_dth'][:, 8])))'''),
('code', "sc = ld('fig2_scans.npz'); make_figs.fig4(min(sc['L3'].min(), sc['L1'].min())); show('paper/figs/fig4_mech.pdf')"),
('md', '**Supplementary Figure (Note 9).** Along $r_e$ the QFI is large and flat while $G_{r_e}$ crosses zero at the basin; along $\\Delta\\theta$ the QFI is three orders smaller yet the phase gradient steers the optimizer. QFI is a budget, not a predictor. Panel d: the quadrature means carry 17% of $S_r^2$ but 1% of $S_{r_e}^2$ and 0.3% of $S_{\\Delta\\theta}^2$.'),
], '03_mechanism_sensitivity_qfi.ipynb')

# ---------------------------------------------------------------------------------------------- 04 cross task
nb('The same controls reach better reservoirs for four tasks; comparison with re-fabrication', [
('md', r'''## Statement of the problem
Hold the fabricated device fixed and optimize the squeezing controls for four tasks. Compare the gain with what **re-fabricating** the device (different $g,\kappa,\epsilon$ at zero squeezing) would give, and read off the effective device parameters at each optimum.

**Tasks.** Ten-step Mackey–Glass; NARMA10 ($y_{t+1}=0.3y_t+0.05y_t\sum_{i=0}^9 y_{t-i}+1.5u_{t-9}u_t+0.1$); two-step Lorenz-63 ($x$ component, RK4, sampled every 0.05); nonlinear channel equalization (Jaeger & Haas 2004, 24 dB SNR, target $d_{n-2}$).

**Protocol per task.** Coarse $6\times6\times12$ scan; projected FD gradient descent from 4 random starts (100 evaluations); refabrication grid $g\in[0.25,0.9]$, $\kappa\in[0.1,0.6]$, $\epsilon\in[0.1,0.3]$ ($6\times6\times5$, extended to $\epsilon=0.5$ in `run_review.py refab`); effective parameters at the unsqueezed and optimized settings. The Lorenz task is drive-limited and is rerun with the opposite parametric phase ($\phi_s=\pi$, input amplified to $\epsilon e^{+r}$) in `run_lorenz_flip.py`.'''),
('code', HEADER),
('md', '## Compute\n`run_tasks.py` (~35 min), `run_lorenz_flip.py` (~5 min), `run_review.py refab` (~5 min).'),
('code', '''if RUN:
    for cmd in (['python3', 'run_tasks.py'], ['python3', 'run_lorenz_flip.py'], ['python3', 'run_review.py', 'refab']): subprocess.run(cmd)
tk = ld('tasks.npz'); lf = ld('lorenz_flip.npz'); rv = ld('review.npz')
for n, name in (('mg', 'Mackey-Glass'), ('narma', 'NARMA10'), ('lorenz', 'Lorenz-63'), ('nce', 'channel equalization')):
    L0 = tk[f'{n}_L3'][0, 0, 0]; Lb = float(tk[f'{n}_bestL']); Lr = tk[f'{n}_refab'].min(); Lr2 = min(Lr, rv[f'refabx_{n}'].min()) if rv is not None and f'refabx_{n}' in rv else Lr
    print('%-22s squeezing %2.0f%% at (%.2f, %.2f, %.2f) | refab %2.0f%% (grid eps<=0.3), %2.0f%% (eps<=0.5) | recovered %s' % (name, 100 * (1 - Lb / L0), *tk[f'{n}_best'], 100 * (1 - Lr / L0), 100 * (1 - Lr2 / L0), '%.0f%%' % (100 * (L0 - Lb) / (L0 - Lr)) if L0 > Lr else 'n/a'))
if lf is not None: L0 = tk['lorenz_L3'][0, 0, 0]; print('Lorenz, amplifying phase: %.0f%% at (%.2f, %.2f, %.2f)' % (100 * (1 - lf['L3'].min() / L0), *lf['best']))'''),
('code', "make_figs.fig3_tasks(); make_figs.efftable(); show('paper/figs/fig3_tasks.pdf'); display(Markdown('```latex\\n' + open('paper/efftable.tex').read() + '\\n```'))"),
('md', '**Figure 3.** (a) Loss relative to unsqueezed: squeezed optimum vs best re-fabricated device. (b) Gradient-descent curves. (c) Optimal controls per task. (d) Held-out NRMSE. The effective-parameter table states each optimum in fabrication terms. Lorenz needs the amplifying branch; the laser (notebook 06) needs the de-amplifying one; the parametric phase is a binary fourth control.'),
], '04_cross_task_and_refabrication.ipynb')

# ---------------------------------------------------------------------------------------------- 05 robustness
nb('Robustness: finite-difference step, measurement noise, Fock cutoff, ridge, extended box, second device, new realizations, baselines', [
('md', r'''## Statement of the problem
Bound the claims of the main text: (i) is the finite-difference gradient stable in the step? (ii) does it survive measurement noise? (iii) is the Fock truncation converged at the optima? (iv) does the landscape depend on the assumed readout precision (ridge)? (v) are the boundary optima box artefacts? (vi) does a device that is already good gain anything? (vii) do the optima transfer to new input realizations? (viii) how does the reservoir compare with reservoir-free regressions and an echo-state network?

Scripts: `run_supp.py {fd,noise,fock,ridge,narma}` and `run_review.py {gap,baselines,box,seeds,refab,device2}`.'''),
('code', HEADER),
('code', '''if RUN:
    for p in ('fd', 'fock', 'noise', 'narma', 'ridge'): subprocess.run(['python3', 'run_supp.py', p])
    for p in ('baselines', 'box', 'seeds', 'device2'): subprocess.run(['python3', 'run_review.py', p])
rv = ld('review.npz'); fd = ld('supp_fd.npz'); nz = ld('supp_noise.npz'); ng = ld('supp_noisygd.npz'); fk = ld('supp_fock.npz'); rd = ld('supp_ridge.npz')
print('FD step (mid-box anchor), gradient at h=1e-3 vs 1e-2:', fd['fd_mid'][fd['fd_h'] == 1e-3][0].round(5), fd['fd_mid'][fd['fd_h'] == 1e-2][0].round(5))
G0, Gn = nz['G0_mid'], nz['Gn_mid']; a = np.argmin(abs(nz['hs'] - 0.1)); b = np.argmin(abs(nz['sig'] - 0.01))
print('sign recovery at 1%% noise, h=0.1 (r, re, dth):', [np.mean(np.sign(Gn[a, b, :, i]) == np.sign(G0[a, i])) for i in range(3)])
print('noisy GD: median noiseless-loss reduction at 1%% noise: %.0f%%' % (100 * np.median(1 - ng['clean_0.01'] / ng['L0_clean'])))
print('Fock cutoff at the basin: L(Nc=8)/L(Nc=20) - 1 = %.1e' % (fk['fock_best_gd'][1, 1] / fk['fock_best_gd'][-1, 1] - 1))
from scipy.stats import spearmanr
print('ridge: rank correlation with 1%% landscape at sigma_rel =', dict(zip(rd['sigs'], [round(spearmanr(rd['L'][1].ravel(), rd['L'][a].ravel()).correlation, 2) for a in range(4)])))
for n in ('mg', 'narma', 'lorenz', 'nce'): print('baselines', n, [(r[0], round(float(r[2]), 3)) for r in rv[f'base_{n}']])
print('second device (T_in=4): best squeezing gain %.1f%%' % (100 * (1 - rv['dev2_L3'].min() / rv['dev2_L3'][0, 0, 0])))
for n in ('mg', 'narma', 'lorenz', 'nce'): S = rv[f'seeds_{n}']; print('transfer to new realizations', n, (100 * (1 - S[:, 1] / S[:, 0])).round(0))'''),
('code', "make_figs.figS1(); make_figs.figS2(ld('fig2_scans.npz')['L3'][0,0,0]); make_figs.figS3(); make_figs.figS4(); make_figs.figS_box(); make_figs.basetable()\nfor f in ('figS1_fd', 'figS2_noise', 'figS3_narma', 'figS4_fock_ridge', 'figS_box'): show(f'paper/figs/{f}.pdf')\ndisplay(Markdown('```latex\\n' + open('paper/basetable.tex').read() + '\\n```'))"),
('md', 'The amplitude gradients survive 1% noise; the phase gradient (10–100× smaller) does not. The ridge moves the basin location with the assumed precision — an argument for in-situ training. A device already good for the task gains nothing. The squeezed reservoir matches linear regression on ten delays and is far from a tuned ESN: the contribution is control, not accuracy.'),
], '05_robustness_and_baselines.ipynb')

# ---------------------------------------------------------------------------------------------- 06 laser
nb('Real-world task I: one-step prediction of the Santa Fe far-infrared laser intensity', [
('md', r'''## Statement of the problem
Experimental chaotic data (Santa Fe competition set A, far-infrared NH$_3$ laser, 10,093 samples; first 1,000 used, Mackey–Glass split, one-step target). The intensity collapses are a strongly nonlinear feature: does the emitter resolve them, and do the squeezing controls help?

**Protocol.** Coarse scan, gradient descent (4 starts), refabrication grid extended to $\epsilon=0.5$, Fock check at the optimum, transfer to the segments starting at samples 2,500 and 5,000, reservoir-free baselines, amplifying-branch scan. Script `run_laser.py` (~15 min); the series comes from `reservoirpy.datasets.santafe_laser`.'''),
('code', HEADER),
('code', '''if RUN: subprocess.run(['python3', 'run_laser.py'])
ls_ = ld('laser.npz'); L0 = ls_['L3'][0, 0, 0]; Lb = float(ls_['bestL'])
print('unsqueezed: L0 = %.1f, NRMSE %.3f;  baselines:' % (L0, ls_['N3'][0, 0, 0]), [(r[0], round(float(r[2]), 3)) for r in ls_['base']])
print('squeezed optimum (%.2f, %.2f, %.2f): L = %.1f (%.0f%%), NRMSE %.3f (Nc=16: %.3f), photons %.2f' % (*ls_['best'], Lb, 100 * (1 - Lb / L0), ls_['conv'][0, 1], ls_['conv'][2, 1], ls_['conv'][2, 2]))
print('refabrication best %.0f%%; recovered %.0f%%; amplifying branch best %.0f%%' % (100 * (1 - ls_['refab'].min() / L0), 100 * (L0 - Lb) / (L0 - ls_['refab'].min()), 100 * (1 - ls_['flipL3'].min() / L0)))
print('transfer to later segments:', (100 * (1 - ls_['seeds'][:, 1] / ls_['seeds'][:, 0])).round(0), '% reductions')'''),
('code', "make_figs.fig_real(); show('paper/figs/fig_real.pdf')"),
('md', '**Figure 5a–c.** Landscape slice, gradient-descent curves, held-out prediction traces. The unsqueezed device already beats linear regression on ten delays; squeezing lowers the loss by 65% (NRMSE 0.31 → 0.18) on the de-amplifying branch — the laser rewards the phase-sensitive emitter response, not a larger drive (Lorenz is the mirror case).'),
], '06_santa_fe_laser.ipynb')

# ---------------------------------------------------------------------------------------------- 07 digits
nb('Real-world task II: spoken-digit recognition (protocol of nonMarkovianReservoirComputer)', [
('md', r'''## Statement of the problem
Automatic spoken-digit recognition on the Free Spoken Digit Dataset with the protocol of the non-Markovian (atom-in-front-of-a-mirror) reservoir study, so that the squeezed cavity is trained on exactly the same task: 5 speakers × 10 digits × 10 recordings; mel spectrogram (`librosa`, `n_fft=len(audio)`, `power_to_db(ref=mean)`) split into a $6\times6$ block grid; block **means** and block **stds** are two input channels of 36 values per utterance, each min–max normalized to $[0,1]$; utterances fed consecutively, no separators; features $=(P,Q)$ at the end of each of the 36 input steps (144 per utterance); 5 washout utterances; ten one-vs-all ridge classifiers ($\delta=10^{-10}$, bias); winner-takes-all; word error rate under 5-fold cross-validation (`KFold(5, shuffle=True, random_state=99)`).

**Selection rule.** $\delta=10^{-10}$ fits the 395 training utterances exactly (training WER 0), so the training loss is a weak surrogate; the operating point is selected by the training loss on fold 1 only, and the 5-fold WER is reported at that point — the test error is never used for selection.

Script `run_asr.py` (features cached in `data/asr_features.npz`; one fold evaluation ≈ 14 s, full pipeline ≈ 40 min). Requires `librosa`, `scikit-learn` and the FSDD recordings in `$FSDD_DIR` (default `~/fsdd`).

Two earlier protocols (`run_digits*.py`: two speakers, 8-channel filterbank time-multiplexed, 192-feature frame-wise readout; then a WTA-aligned cross-validated loss) gave a landscape with a few percent of relief and no resolvable accuracy change at a 60-utterance test set; they are superseded by the protocol above and are kept in the code for the record.'''),
('code', HEADER),
('code', '''if RUN: subprocess.run(['python3', 'run_asr.py', 'all'])
asr = ld('asr.npz')
print('5-fold WER: frontend ridge %s -> %.1f%%; unsqueezed %s -> %.1f%%; squeezed (%.1f, %.1f, %.2f, phi_s=%s) %s -> %.1f%%' % (
    asr['frontend'].round(2), 100 * asr['frontend'].mean(), asr['unsq5'][:, 1].round(2), 100 * asr['unsq5'][:, 1].mean(), *asr['best'], 'pi' if float(asr['best_s']) < 0 else '0', asr['best5'][:, 1].round(2), 100 * asr['best5'][:, 1].mean()))
S = asr['scan']; print('fold-1 training loss: unsqueezed %.5f; best on de-amplifying grid %.5f; best on amplifying grid %.5f' % (S[0, 0, 0], asr['grid_p'][:, :, 0].min(), asr['grid_m'][:, :, 0].min()))
print('six-observable readout (432 features) overfits: WER %.0f%% / %.0f%%' % (100 * asr['unsq5_all'][:, 1].mean(), 100 * asr['best5_all'][:, 1].mean()))'''),
('code', '''import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(11, 3.2)); g = asr['grid']
for ax, tag, tt in zip(axs[:2], ('p', 'm'), ('de-amplifying branch, $\\\\phi_s=0$', 'amplifying branch, $\\\\phi_s=\\\\pi$')):
    G = asr[f'grid_{tag}']; im = ax.pcolormesh(g, g, G[:, :, 0], cmap='viridis_r', shading='nearest'); fig.colorbar(im, ax=ax, label='fold-1 training loss')
    for i in range(6):
        for j in range(6): ax.text(g[j], g[i], '%.0f' % (100 * G[i, j, 1]), ha='center', va='center', fontsize=7, color='w')
    ax.set_xlabel('$r_e$'); ax.set_ylabel('$r$'); ax.set_title(tt + ' (numbers: fold-1 WER %)')
ax = axs[2]; x = np.arange(5); w = 0.27
ax.bar(x - w, 100 * asr['frontend'], w, label='frontend ridge'); ax.bar(x, 100 * asr['unsq5'][:, 1], w, label='unsqueezed'); ax.bar(x + w, 100 * asr['best5'][:, 1], w, label='squeezed')
ax.set_xticks(x); ax.set_xticklabels(['fold %d' % (i + 1) for i in x]); ax.set_ylabel('WER (%)'); ax.legend(frameon=False); ax.set_title('5-fold word error rate')
plt.tight_layout(); plt.savefig('/tmp/asr_nb.png', dpi=120); plt.close(fig); display(Image('/tmp/asr_nb.png'))
make_figs.fig_real(); show('paper/figs/fig_real.pdf')'''),
('md', '**Result.** The fabricated device does no better than its spectrogram frontend (12.0% vs 11.0% WER). The amplifying parametric branch raises the input to $\\epsilon e^{r}$ and the emitter excitation from 0.19 to 2.0 photons; at $(r,r_e,\\Delta\\theta)=(0.4,0,0)$, $\\phi_s=\\pi$, the 5-fold WER is 8.0% — a 33% relative reduction, below the frontend. Same mechanism as Lorenz; opposite branch to the laser.'),
], '07_spoken_digits.ipynb')

# ---------------------------------------------------------------------------------------------- 08 base-first
nb('Optimize the base device first, then squeeze; convergent derivatives', [
('md', r'''## Statement of the problem
The drive amplitude $\epsilon$ and phase $\phi_d$ are set by the drive laser, so they are free parameters of the base device, not fabrication parameters; with the drive phase the effective input strength is $\epsilon_{\rm eff}=\epsilon\sqrt{\cosh 2r-\sinh 2r\cos 2\phi_d}$ (from $\epsilon e^{-r}$ at $\phi_d=0$ to $\epsilon e^{+r}$ at $\phi_d=\pi/2$). The fair question is therefore: **does squeezing improve a device whose drive has already been optimized?**

**Protocol (`run_base.py`).** For each task: (1) scan $\epsilon\in[0.05,1]$ at zero squeezing and keep the best base within a photon budget $\max_t\langle a^\dagger a\rangle\le2.5$ (same budget for the squeezed device; $N_c=12$, checked at 16); (2) scan $(r,r_e,\Delta\theta)$ at $\phi_d=0$ and $(r,r_e)$ grids at $\phi_d=\pi/2$ from that base, masking points over the budget; (3) refine with the convergent gradient. Without the budget the search runs to the truncation limit: the first attempt gave a spurious $-26\%$ on Mackey–Glass at 10–12 photons with a 24% cutoff dependence.

**Convergent derivatives (`common.fd_gradient_conv`).** Richardson extrapolation of central differences at $h$ and $h/2$ with an error estimate, step halved until the error is below 5% of the derivative, second-order one-sided differences inward at box faces (the earlier projected estimator was low by 2–3.4× there). Verified in `run_deriv.py`.'''),
('code', HEADER),
('code', '''if RUN: subprocess.run(['python3', 'run_base.py']); subprocess.run(['python3', 'run_deriv.py'])
b = ld('base.npz')
for n, name in (('mg', 'Mackey-Glass'), ('narma', 'NARMA10'), ('lorenz', 'Lorenz-63'), ('nce', 'channel eq.'), ('laser', 'Santa Fe laser')):
    E = b[f'{n}_eps']; L02 = E[np.isclose(E[:, 0], 0.2)][0, 1]; Lb = E[:, 1].min(); c = b[f'{n}_conv']
    print('%-16s base eps* = %.2f (%3.0f%% vs eps=0.2) | squeezing on the base: %3.0f%% at (%.2f, %.2f, %.2f), phi_d = %s | photons %.1f, cutoff change %.1f%%' % (name, b[f'{n}_epsbest'], 100 * (1 - Lb / L02), 100 * (1 - c[0, 0] / Lb), *b[f'{n}_best'], 'pi/2' if b[f'{n}_bestpd'] > 0.5 else '0', c[1, 2], 100 * abs(c[0, 0] - c[1, 0]) / c[1, 0]))
dv = ld('deriv.npz')
for r in dv['rows']: print('%-16s fixed %s | convergent %s +- %s | reference %s' % (r[0], np.round(r[2], 6), np.round(r[3], 6), np.round(r[4], 7), np.round(r[6], 6)))'''),
('code', "make_figs.fig_base(); show('paper/figs/fig_base.pdf')"),
('md', 'Where the fixed-drive gain was an input-strength gain, the optimized drive takes it (Mackey–Glass: +1% left). Where it comes through the emitter — strong external squeezing at small $r$ (NARMA10, Lorenz, NCE) — or through the detuned cavity and counter-rotating coupling (laser, $r\\approx0.5$) — squeezing improves even the drive-optimized device by 6–67%. These are the parameters no drive setting reaches.'),
], '08_base_first_and_convergent_derivatives.ipynb')

print('notebooks written')

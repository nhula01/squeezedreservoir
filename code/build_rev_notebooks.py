"""Build the three revision notebooks (09-11) from the revision scripts and saved data. Paths are relative to the bundle."""
import nbformat as nbf, os
HERE = os.path.dirname(os.path.abspath(__file__)); NB = os.path.join(HERE, '..', 'notebooks')
HEADER = '''import os, sys, subprocess, numpy as np
CODE = os.path.abspath(os.path.join(os.getcwd(), '..', 'code')); sys.path.insert(0, CODE); os.chdir(CODE); os.environ['OMP_NUM_THREADS'] = '1'
import matplotlib; matplotlib.use('Agg')
from IPython.display import Image, display
RUN = False   # set True to recompute the data of this notebook (cost given in the Compute section)
def show(pdf, dpi=110):
    base = '/tmp/' + os.path.basename(pdf)[:-4]; subprocess.run(['pdftoppm', '-r', str(dpi), '-png', '-singlefile', pdf, base], check=True); display(Image(filename=base + '.png'))
ld = lambda n: np.load(os.path.join('data', n), allow_pickle=True)
'''
def nb(title, cells, fname):
    n = nbf.v4.new_notebook(); n.cells = [nbf.v4.new_markdown_cell(f'# {title}'), nbf.v4.new_code_cell(HEADER)]
    for kind, src in cells: n.cells.append(nbf.v4.new_markdown_cell(src) if kind == 'md' else nbf.v4.new_code_cell(src))
    nbf.write(n, os.path.join(NB, fname))

nb('Full re-fabrication baseline, tuned drive and mapped effective device (referee concern 1)', [
('md', r'''## Problem
The referee noted that the original re-fabrication grid, $(g,\kappa,\epsilon)$ at fixed detuning and emitter decay, does not optimize all parameters that the squeezing controls mimic. The controls move the effective cavity frequency $\Omega_a=\omega_a/\cosh 2r$, the couplings $g\cosh r$ and $g\sinh r$, the input strength $\epsilon_{\rm eff}$ and the emitter relaxation. Three questions:
1. What does a device re-fabricated in **all six** parameters $(g,\kappa,\gamma,\omega_a,\omega_q,\epsilon)$ achieve?
2. The detunings are measured from the drive carrier, so the **drive frequency** is a post-fabrication knob. What does squeezing add to a base whose drive amplitude *and* frequency are optimized?
3. How much of that squeezing gain would an **ordinary device with the same effective parameters** reproduce (mapped effective device)?

## Method
* `rev_refab.py`: Latin hypercube (119 points + the amplitude-optimized base) and three bounded Nelder–Mead refinements in normalized (log for rates) coordinates, 300 evaluations (460 for Mackey–Glass), photon budget $\max_t\langle a^\dagger a\rangle\le 2.5$ by penalty, $N_c=10$, optimum checked at $N_c=14$.
* `rev_base2.py`: $(\epsilon,\delta)$ grid + Nelder–Mead at zero squeezing, then $3\times3\times8$ scans of $(r,r_e,\Delta\theta)$ for $\phi_d\in\{0,\pi/2\}$ and 60 evaluations of convergent-gradient descent; cutoff check at $N_c+4$.
* `rev_post.py`: mapped effective device, bare $\gamma$ found by bisection so that the mean $\gamma^{\rm eff}_{x,y,z}$ matches the squeezed device.'''),
('md', '## Compute (≈ 35 min for re-fabrication, ≈ 25 min for the tuned base, 1 core)'),
('code', "if RUN:\n    for s in ('rev_refab.py', 'rev_base2.py', 'rev_post.py'): subprocess.run([sys.executable, s], check=True)"),
('md', '## Results'),
('code', '''rf, b2, po, bs = ld('rev_refab.npz'), ld('rev_base2.npz'), ld('rev_post.npz'), ld('base.npz')
pc = lambda a, b: 100 * (1 - a / b)
print('%-8s %10s %10s %10s %10s %10s %10s' % ('task', 'amp base', 'tuned', 'squeezed', 'mapped', 'refab', 'sq/refab'))
for n in ['mg', 'narma', 'lorenz', 'nce', 'laser']:
    Lb = bs[f'{n}_eps'][:, 1].min(); Lt = b2[f'{n}_base2'][2]; Ls = b2[f'{n}_Lsq2'][0][0]; Lm = po[f'{n}_map2_Lfull'][0]; Lr = rf[f'{n}_best'][6]
    print('%-8s %10.3e %10.3e %10.3e %10.3e %10.3e %10.2f' % (n, Lb, Lt, Ls, Lm, Lr, Ls / Lr))
    print('         tuned vs amp %+.0f%%  squeezing vs tuned %+.0f%%  mapped vs tuned %+.0f%%  refab vs tuned %+.0f%%' % (-pc(Lt, Lb), -pc(Ls, Lt), -pc(Lm, Lt), -pc(Lr, Lt)))'''),
('code', "subprocess.run([sys.executable, 'make_rev.py'], check=True, capture_output=True); show('../paper/figs/fig_refab.pdf')"),
('code', "print(open('../paper/refabtable.tex').read())"),
('md', r'''## Interpretation
* The drive frequency alone captures most of what squeezing appeared to give against a fixed drive on Lorenz and the laser; those were detuning gains.
* Over the fully tuned drive, squeezing adds a gain on NARMA10, Lorenz and channel equalization that the mapped ordinary device does **not** reproduce: it comes from the counter-rotating coupling $g\sinh r$ and the anisotropic, phase-sensitive noise of the squeezed bath.
* Full re-fabrication beats the squeezed device on four of five tasks (NARMA10 is the exception), with several optima at the search bounds. The claim that squeezing recovers or exceeds the re-fabrication gain is withdrawn.'''),
], '09_full_refabrication.ipynb')

nb('Experimental cost of in-situ training (referee concern 2, cost)', [
('md', r'''## Problem
How many repetitions of the experiment does a loss evaluation need, does the squeezing gain survive realistic readout noise, how accurate are finite-difference gradients at a given budget, and can the controls be trained from measured losses?

## Readout model (`expt.py`)
Time-resolved homodyne of the cavity output (rate $\kappa$, efficiency $\eta_c=0.8$) and of the emitter fluorescence (rate $\gamma$, $\eta_e=0.5$). One pass of the sequence with the local oscillators at $(Q,\sigma_x)$ or $(P,\sigma_y)$ yields one estimate of every feature at every virtual time; a loss evaluation costs $2N_{\rm rep}$ passes. Per bin $\tau=T_{\rm in}/N_v$ the detector adds variance $s=1/(\eta\,\text{rate}\,\tau)$:
$$\mathrm{Var}[\hat Q]=\mathrm{Var}(Q)+s_c,\quad \mathrm{Var}[\widehat{Q^2}]=\mathrm{Var}(Q^2)+4\langle Q^2\rangle s_c+2s_c^2,\quad \mathrm{Cov}=\langle Q^3\rangle-\langle Q\rangle\langle Q^2\rangle+2\langle Q\rangle s_c,\quad \mathrm{Var}[\hat\sigma_x]=1-\langle\sigma_x\rangle^2+s_e.$$
Physical units: $\kappa/2\pi=10$ MHz, $\omega_a=5\kappa$, one pass of 1000 symbols = 6.4 µs.

## Studies
* `rev_calib.py` / `rev_post.py`: realized precision and measured squeezing gain vs $N_{\rm rep}$.
* `rev_grad.py`: gradient error and sign recovery vs step $h$ and $N_{\rm rep}$; $h^\star=(3\sigma_{\mathcal L}/|\mathcal L^{(3)}|)^{1/3}$.
* `rev_train.py`: finite-difference gradient descent and SPSA on measured losses, 48 evaluations, from the tuned base with squeezers off.'''),
('md', '## Compute (≈ 3 min calibration, ≈ 2 min gradients, ≈ 17 min training)'),
('code', "if RUN:\n    for s in ('rev_post.py', 'rev_grad.py', 'rev_train.py'): subprocess.run([sys.executable, s], check=True)"),
('md', '## Results'),
('code', '''po = ld('rev_post.npz')
for n in ['mg', 'narma', 'lorenz', 'nce', 'laser']:
    rb, rs = po[f'{n}_base'], po[f'{n}_sq']
    print(n.ljust(7), 'sigma_rel(median) at N=1e6: %.3f' % rs[3, 1], '| measured gain:', ' '.join('%.0e:%+.0f%%' % (N, -100 * (1 - rs[i, 8] / rb[i, 8])) for i, N in enumerate(po['NREP'])))'''),
('code', '''gr = ld('rev_grad.npz'); print('task', gr['task'])
for an in ('start', 'mid'):
    g = gr[f'{an}_gtrue']; print(an, 'grad', g.round(4), 'sigma_L/L', (gr[f'{an}_sigL'] / gr[f'{an}_L0']).round(4), "L3", gr[f'{an}_L3'].round(3))
    for b, N in enumerate(gr['NREP']):
        G = gr[f'{an}_G'][:, b]; err = np.sqrt(np.mean(np.sum((G - g) ** 2, -1), -1)) / np.linalg.norm(g)
        print('   N=%.0e  rel. rms error vs h' % N, dict(zip(gr['H'], err.round(2))))'''),
('code', '''tr = ld('rev_train.npz'); print('task', tr['task'])
for k in sorted(f for f in tr.files if f.endswith('_Lfinal')): print(k, '%.4e' % float(tr[k]), 'passes %.1e' % float(tr[k.replace('Lfinal', 'passes')]))'''),
('md', '## Noise-aware readout (negative result) and pre-scan training (`rev_snr.py`, `rev_train2.py`)'),
('code', '''sn = ld('rev_snr.npz')
for n in ['narma', 'lorenz', 'nce', 'laser']:
    rb, rs = sn[f'{n}_base'], sn[f'{n}_sq']
    print(n.ljust(7), 'gain protocol / noise-aware readout vs N:', ' '.join('%.0e:%+.0f/%+.0f%%' % (N, -100 * (1 - rs[i, 1] / rb[i, 1]), -100 * (1 - rs[i, 3] / rb[i, 3])) for i, N in enumerate(sn['NREP'])))
print('loss noise (protocol, noise-aware) at the start anchor for N=1e5,1e6,1e7:', sn['sigL_start'][:, :2].round(5))
t2 = ld('rev_train2.npz')
for k in sorted(f for f in t2.files if f.endswith('_Lfinal')): print(k, '%.4e' % float(t2[k]))'''),
('md', r'''## Squeezed input noise at the detector (`rev_lo.py`, `rev_lo_diag.py`)
The reflected squeezed input sets the homodyne white noise: $s_c(\phi)=[\eta_c V_{\rm in}(\phi)+1-\eta_c]/(\eta_c\kappa\tau)$, $V_{\rm in}(\phi)=1+2N-2\,\mathrm{Re}(M^*e^{-2i\phi})$. Does choosing the LO angle lower the cost?'''),
('code', "import expt; expt.selftest_lo(); print(open('../paper/lotable.tex').read())"),
('code', '''dg = ld('rev_lo_diag.npz')
for n in ['narma', 'lorenz', 'nce', 'laser']:
    r = dg[f'{n}_QP']; L0 = dg[f'{n}_QP_L0']; i = list(dg['NN']).index(1e6)
    print(n.ljust(7), 'V_in(Q,P,sq,anti)', dg[f'{n}_V'].round(2), '| L/L0 at 1e6: all noise %.2f, emitter noiseless %.2f' % (r[i, 1] / L0[0], r[i, 2] / L0[0]))'''),
('md', 'Conclusion: squeezed detection does not lower the cost here: the squeezing axis is not aligned with the informative quadratures, a single squeezed quadrature discards information, and the emitter channel dominates the noise.'),
('code', "subprocess.run([sys.executable, 'make_rev.py'], check=True, capture_output=True); show('../paper/figs/fig_cost.pdf')"),
('md', r'''## Interpretation
* The original "$10^4$ shots for 1 % precision" was wrong by two to three orders of magnitude: the detector noise per bin dwarfs the input-driven feature variation.
* Squeezing gains are measurable only above a task-dependent budget ($10^5$–$10^6$ passes per setting for channel equalization, more for Lorenz and the laser).
* The loss noise has a floor from standardizing noise-dominated features; weighting features by measured SNR would remove it.
* Training from measured losses reaches 60–95 % of the noiseless gain in most runs (minutes to an hour of measurement per run at $\kappa/2\pi=10$ MHz), but some runs stall near the starting corner; restarts or a coarse pre-scan are needed.'''),
], '10_experimental_cost.ipynb')

nb('Non-ideal squeezing and drift (referee concern 2, non-idealities)', [
('md', r'''## Problem
Does the squeezing gain survive injection loss of the external squeezing, phase jitter relative to the pump, slow drift of the controls and drive, and the readout — separately and together?

## Model (`expt.py`)
Lossy, phase-diffused squeezed bath: $\kappa(N+1)\mathcal D[a]+\kappa N\mathcal D[a^\dagger]+\kappa M\mathcal S[a^\dagger]+\kappa M^*\mathcal S[a]$ with $N=\eta_{\rm inj}\sinh^2 r_e$, $M=\eta_{\rm inj}\cosh r_e\sinh r_e e^{i\theta}e^{-\sigma_\phi^2/2}$ (reduces exactly to $\mathcal D[L_a]$ for $\eta_{\rm inj}=1$, $\sigma_\phi=0$; `expt.selftest()`). Drift: per-evaluation relative errors on $r$, $r_e$, $\epsilon$ and absolute errors on $\Delta\theta$, $\phi_d$ (moderate 1 %/0.02 rad, large 3 %/0.05 rad), applied to the base device as well.'''),
('code', "import expt; expt.selftest()"),
('md', '## Compute (≈ 13 min, `rev_nonideal.py`)'),
('code', "if RUN: subprocess.run([sys.executable, 'rev_nonideal.py'], check=True)"),
('md', '## Results'),
('code', '''ni = ld('rev_nonideal.npz'); pc = lambda a, b: 100 * (1 - a / b)
for n in ['mg', 'narma', 'lorenz', 'nce', 'laser']:
    if f'{n}_comb' not in ni.files: continue
    Lb = ni[f'{n}_Lb']
    print(n.ljust(7), 'ideal %.0f%%' % pc(ni[f'{n}_Ls'], Lb), '| eta 1,.9,.7,.5:', np.round(pc(ni[f'{n}_eta'], Lb)), 'reopt(.7) %.0f%%' % pc(ni[f'{n}_reopt_7'][4], Lb),
          '| jitter .05-.3:', np.round(pc(ni[f'{n}_sig'], Lb)), '| drift mod/large %.0f/%.0f%%' % tuple(pc(ni[f'{n}_drift_{d}'][:, 0].mean(), ni[f'{n}_drift_{d}'][:, 1].mean()) for d in ('moderate', 'large')),
          '| readout 1e6 %.0f%% | all %.0f%%' % (pc(ni[f'{n}_ro'][:, 0].mean(), ni[f'{n}_ro'][:, 1].mean()), pc(ni[f'{n}_comb'][:, 0].mean(), ni[f'{n}_comb'][:, 1].mean())))'''),
('code', "subprocess.run([sys.executable, 'make_rev.py'], check=True, capture_output=True); show('../paper/figs/fig_nonideal.pdf')"),
('md', r'''## Interpretation
Injection loss down to $\eta_{\rm inj}=0.5$, phase jitter up to 0.3 rad and drift barely change the gain; the readout is the binding constraint.'''),
], '11_nonidealities.ipynb')
nb('Spoken digits against a tuned drive', [
('md', r'''## Problem
Does squeezing improve spoken-digit recognition once the drive amplitude and frequency are tuned?
## Method (`rev_asr2.py`, protocol of `run_asr.py`, features from `data/asr_features.npz`)
Operating points selected by the first-fold training loss only: drive grid eps in {0.1, 0.2, 0.4} x delta in {-0.5..0.5}; squeezing scan from the tuned base, (r, r_e) in {0, 0.25, 0.5}^2, dth in {0, pi}, phi_d in {0, pi/2}; photon budget 2.5; five-fold WER at the selected points. Compute ~30 min.'''),
('code', "if RUN: subprocess.run([sys.executable, 'rev_asr2.py'], check=True)"),
('code', '''a = ld('rev_asr2.npz'); W = a['wer5']
for lab, w in zip(['fabricated', 'tuned drive', 'tuned + squeezed'], W): print(lab.ljust(18), 'WER per fold', (100 * w[:, 1]).round(0), 'mean %.1f%%' % (100 * w[:, 1].mean()))
print('selected (eps, delta, phi_d, r, r_e, dth):', a['sel'].round(3))
S = a['scan']; i = S[:, 4].argmin(); print('lowest scanned loss', S[i, 4], 'at', S[i, :4].round(3), 'nmax %.2f (budget 2.5)' % S[i, 7])'''),
('md', 'The WER gain is a drive-tuning gain; squeezing the tuned device changes the WER by less than the fold resolution.'),
('md', '''## Control: drive amplitude only (`run_asr_base.py`)
Holding the drive frequency at its fabricated value and scanning only the amplitude isolates which part of the drive matters. Amplitude alone is inert; squeezing from the amplitude-only base helps only through intracavity squeezing, whose cavity frequency $\\Omega_a=\\omega_a/\\cosh 2r$ is the detuning the tuned drive supplies directly.'''),
('code', '''b = ld('asr_base.npz')
print('amplitude scan at zero squeezing (eps, fold-1 loss, fold-1 WER):', [(round(float(r[0]), 2), round(float(r[1]), 5), round(float(r[2]), 2)) for r in b['eps']])
print('5-fold WER: amplitude-optimized base (eps=%.2f) %.1f%% -> squeezed on that base (r, re)=(%.2f, %.2f), phi_d=pi/2: %.1f%%' % (b['epsbest'], 100 * b['base5'][:, 1].mean(), *b['best'][:2], 100 * b['best5'][:, 1].mean()))
print('cavity frequency at the squeezed optimum: Omega_a = %.2f  (tuned drive detuning in rev_asr2: delta = %.2f)' % (1 / np.cosh(2 * b['best'][0]), a['sel'][1]))'''),
], '12_spoken_digits_tuned_drive.ipynb')
print('built')

"""Build notebook 13 (second review round: wide-box re-fabrication and joint search, fresh realizations, emitter detection
efficiency, quantum-trajectory check of the readout model, input-phase convention) and execute it."""
import nbformat as nbf, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); NB = os.path.join(HERE, '..', 'notebooks')
HEADER = '''import os, sys, subprocess, numpy as np
CODE = os.path.abspath(os.path.join(os.getcwd(), '..', 'code')); sys.path.insert(0, CODE); os.chdir(CODE); os.environ['OMP_NUM_THREADS'] = '1'
import matplotlib; matplotlib.use('Agg')
from IPython.display import Image, display
RUN = False   # set True to recompute the data of this notebook (cost given in each Compute section)
def show(pdf, dpi=110):
    base = '/tmp/' + os.path.basename(pdf)[:-4]; subprocess.run(['pdftoppm', '-r', str(dpi), '-png', '-singlefile', pdf, base], check=True); display(Image(filename=base + '.png'))
ld = lambda n: np.load(os.path.join('data', n), allow_pickle=True)
pc = lambda a, b: 100 * (1 - a / b)
TASKS = ['mg', 'narma', 'lorenz', 'nce', 'laser']'''

cells = [
('md', r'''## Questions of the second review
1. **Boundary artefact.** Several re-fabricated optima of `rev_refab.py` touch the search box (NARMA10 $g=1.20$, Lorenz $\epsilon=1.596$, channel equalization $\kappa=0.99$, Mackey–Glass $\omega_a,\omega_q=2.0$), and squeezing rescales $g_{\rm co}=g\cosh r$ beyond the face (NARMA10: $1.20\cosh0.241=1.235$). Is the gain of squeezing on the re-fabricated device a gain of squeezing, or of leaving the box?
2. **Seeds.** Do the tuned-drive headline gains (NARMA10, channel equalization) survive fresh input realizations?
3. **Emitter detection efficiency.** The emitter channel dominates the readout cost; how do the conclusions change for $\eta_e=0.3, 0.1$?
4. **Readout model.** The cost section uses a bin model (single-time variance + white detector noise, within-bin input–field correlation neglected). Is it right?'''),
('md', r'''## 1. Wide-box re-fabrication, squeezing on it, and a joint search (`rev_bounds.py`, `rev_bounds2*.py`)
`rev_bounds.py` runs the study below with the paper's noiseless objective; its ordinary stage (NARMA10, channel equalization) exposed the weak-signal problem, and the study was completed with the measurement-aware objective in an even wider box by `rev_bounds2.py` ($g\in[0.005,3]$, $\kappa\in[0.005,3]$, $\gamma\in[0.002,2]$, $\omega_{a,q}\in[0,5]$, $\epsilon\in[0.001,5]$; ordinary 506 evaluations; sequential = 300 + 206; joint 506), `rev_bounds2_final.py` (squeezing on the converged ordinary optimum), `rev_bounds2_jointzero.py`, `rev_bounds2_jointlocal.py` and `rev_bounds2_budget.py` (Lorenz, photon budget 4). The stages as first designed:
Per task, with photon budget 2.5 and $N_c=10$ (optima checked at $N_c=14$, and $18$ near the budget):
* **sanity check** – old re-fabricated optimum with $g\to g\cosh r_{\rm sq}$ and the drive $(\epsilon\le5,\delta)$ re-tuned, no squeezing;
* **ordinary** – $(g,\kappa,\gamma,\omega_a,\omega_q,\epsilon)$ in $g\in[0.02,3]$, $\kappa\in[0.01,3]$, $\gamma\in[0.005,2]$, $\omega_{a,q}\in[0,3]$, $\epsilon\in[0.01,5]$; 200-point Latin hypercube + the old optimum, Nelder–Mead from the three best distinct points, continuation; 706 evaluations;
* **sequential** – squeezing search of Sec. 2.4 ($2\times72$ scan + 60 convergent-descent evaluations) from the ordinary optimum after 500 evaluations: 706 evaluations in total, the same as *ordinary*;
* **matched** – squeezing on the ordinary optimum with $\Omega_a$, $g\cosh r$ and $\epsilon_{\rm eff}$ pinned to their unsqueezed values ($\omega_a\to\omega_a\cosh2r$, $g\to g/\cosh r$, $\epsilon\to\epsilon/\sqrt{\cosh2r-\sinh2r\cos2\phi_d}$): only $g\sinh r$ and the phase-sensitive damping can act;
* **mapped** – unsqueezed device with the effective parameters of the sequential optimum, then Nelder–Mead over all six parameters (120 evaluations);
* **joint** – all ten parameters $(g,\kappa,\gamma,\omega_a,\omega_q,\epsilon,r,r_e,\Delta\theta,\phi_d)$ from scratch, same protocol and budget (706) as *ordinary*.'''),
('md', '### Compute (≈ 45 min per task on one core; run two tasks in parallel)'),
('code', "if RUN:   # ~45 min per task and stage set on one core; run tasks in parallel processes\n    for s in (['rev_bounds.py', 'narma', 'nce'], ['rev_bounds2.py', 'narma', 'lorenz', 'mg'], ['rev_bounds2.py', 'nce', 'laser'],\n              ['rev_bounds2_final.py', 'mg', 'narma', 'lorenz', 'nce', 'laser'], ['rev_bounds2_jointzero.py', 'mg', 'narma', 'lorenz', 'nce', 'laser'],\n              ['rev_bounds2_jointlocal.py', 'nce'], ['rev_bounds2_budget.py']):\n        subprocess.run([sys.executable, *s], check=True)"),
('md', '### Results'),
('code', r'''PN = ['g', 'kappa', 'gamma', 'omega_a', 'omega_q', 'eps']
for n in ('narma', 'nce'):                                   # noiseless objective, wide box: where the optimum goes
    d = ld(f'rev_bounds_{n}.npz'); T = d['ord_trace']; ok = T[:, 12] <= 2.5; b = T[ok][T[ok, 10].argmin()]
    print(f'{n}: noiseless objective, wide box ->', dict(zip(PN, b[:6].round(4))), 'L %.4e, max photons %.1e' % (b[10], b[12]))
print()
for n in TASKS:                                              # measurement-aware objective
    d = ld(f'rev_bounds2_{n}.npz'); q = ld(f'rev_bounds2_{n}_sqf.npz'); c = lambda k: d[k][-1][0]
    o, j = d['ord_best'], d['joint_best']; R = d['refs'][:, 0]
    print(f'--- {n}: tuned base {R[0]:.4e}, tuned+sq {R[1]:.4e}, old-box refab {R[2]:.4e}, old-box refab+sq {R[3]:.4e}')
    print('  wide-box ordinary', dict(zip(PN, o[:6].round(3))), 'nmax %.2f  L %.4e (checked %.4e)' % (o[12], o[10], c('chk_ord')))
    print('  squeezing on it: theta', q['best'][6:9].round(3), 'phi_d %.2f  L %.4e (checked %.4e, %+.2f%%)' % (q['best'][9], q['best'][10], q['chk'][-1][0], -pc(q['chk'][-1][0], q['chk0'][-1][0])))
    print('  matched %.4e   sequential (300+206) %.4e   joint %.4e (%+.1f%%)' % (d['mat_best'][10], d['seq_best'][10], j[10], -pc(c('chk_joint'), c('chk_ord'))))
jz, jl = ld('rev_bounds2_nce_jointzero.npz'), ld('rev_bounds2_nce_jointlocal.npz')
print('\nchannel eq.: joint device unsqueezed %.4e; ordinary re-optimized from it %.4e' % (jz['L'][0], jl['best'][10]))
bu = ld('rev_bounds2_lorenz_budget.npz'); print('Lorenz, photon budget 4: ordinary %.4e, squeezed (best of two starts)' % bu['ord_best'][10], bu['sq'][:, 4], 'N_c=20:', bu['chk20'][:, 0])'''),
('code', "subprocess.run([sys.executable, 'make_rev.py'], check=True, capture_output=True)\nshow('../paper/figs/fig_refab.pdf'); show('../paper/figs/figS_bounds.pdf')"),
('md', 'For reference, squeezing on the re-fabricated optimum of the **original** box (`rev_refab_sq.py`, `rev_refab_sq14.py`), the study the wide box supersedes:'),
('code', "print(open('../paper/refabsqtable.tex').read())"),
('md', r'''## 2. Fresh realizations (`rev_seeds.py`, ≈ 35 min)
Eight new realizations (seeds 1–8) of NARMA10 and channel equalization. (a) The paper's tuned base and squeezed setting evaluated unchanged; (b) the drive re-tuned (30 Nelder–Mead evaluations) and the squeezing controls re-optimized (60 convergent-descent evaluations) on each realization.'''),
('code', "if RUN: subprocess.run([sys.executable, 'rev_seeds.py'], check=True)\nprint(open('../paper/seedtable.tex').read())\nsd = ld('rev_seeds.npz')\nfor n in ('narma', 'nce'):\n    R = np.array([sd[f'{n}_{s}_reopt'] for s in range(1, 9) if f'{n}_{s}_reopt' in sd.files])\n    print(n, 'gain per seed (%):', (100 * R[:, 3]).round(1))"),
('md', r'''## 3. Emitter detection efficiency (`rev_etae.py`, ≈ 2 min)
The measured-gain calculation of Fig. 3a repeated for $\eta_e=0.5,0.3,0.1$. Last column: factor by which the passes needed for a given median precision grow (median over the ten operating points).'''),
('code', "if RUN: subprocess.run([sys.executable, 'rev_etae.py'], check=True)\nprint(open('../paper/etatable.tex').read())"),
('md', r'''## 4. Quantum trajectories of the homodyne currents (`rev_sme.py`, ≈ 15 min)
**Exact vacuum representation of the squeezed bath.** $\mathcal D[L_a]$ with $L_a=\sqrt\kappa(\mu a+\nu a^\dagger)$ is generated by a vacuum field $b_0$ through $b_{\rm in}=\mu b_0-\nu b_0^\dagger$; then $b_{0,\rm out}=\mu b_{\rm out}+\nu b_{\rm out}^\dagger=b_0+L_a$, and homodyne detection of the physical quadrature $x_\Phi(b_{\rm out})$ is homodyne detection of $x_{\Phi'}(b_{0,\rm out})$ scaled by $A=\sqrt{V_{\rm in}(\Phi)}$ with $A\,x_{\Phi'}(L_a)=\sqrt\kappa\,x_\Phi(a)$; efficiency $\eta_c$ becomes $\eta'=\eta_cV_{\rm in}/(\eta_cV_{\rm in}+1-\eta_c)$.

**Simulation.** Stochastic Schrödinger equation with both channels unravelled by ideal homodyne detection (Itô–Euler, $dt=0.005$, check at $dt/2$); inefficiency added to the currents as independent white noise. 8,000 trajectories from pure states sampled from the unconditional state at symbol 300, 12 symbols. Estimators $\hat X=AJ_c/(\sqrt\kappa\tau)$, $\widehat{X^2}=\hat X^2-s_c$, $\hat\sigma=J_e/(\sqrt\gamma\tau)$, compared with the bin model per bin; then the loss noise of the full readout with covariances rescaled to the trajectory values.'''),
('code', "if RUN: subprocess.run([sys.executable, 'rev_sme.py'], check=True)\nprint(open('../paper/smetable.tex').read())\nshow('../paper/figs/figS_sme.pdf')\ns = ld('rev_sme.npz')\nfor k in ('base', 'sq'):\n    print(k, 'clean L / bin-averaged', s[f'{k}_L0'], ' rows [N, model mean, model std, traj-cal mean, traj-cal std]:'); print(s[f'{k}_loss'])"),
('md', r'''## 5. Input-phase convention of the squeezed-detection study
The first revision used $\langle b_{\rm in}^2\rangle=-M^*$ in `expt.input_variance`. Input–output theory for $L_a=\sqrt\kappa(\mu a+\nu a^\dagger)$ gives $\langle b_{\rm in}^2\rangle=-\mu\nu=-M$, which is also $\langle a^2\rangle$ of the bath-only steady state (`expt.check_input_phase()`). The two agree for the $Q$ and $P$ settings used everywhere in the main text and differ only for rotated local oscillators; `rev_lo.py` and `rev_lo_diag.py` were rerun.'''),
('code', "import expt; expt.check_input_phase()\nprint(open('../paper/lotable.tex').read())"),
('md', "## Interpretation\n* **Boundary artefact, confirmed.** With the paper's noiseless objective a wider box sends the optimum to vanishing drive (NARMA10, $\\epsilon=0.01$, $\\max\\langle a^\\dagger a\\rangle\\approx10^{-4}$) or a decoupled emitter (channel equalization): a relative 1 % precision makes a vanishing signal free. With the measurement-aware objective ($10^7$ passes, noise-matched ridge) the wide-box re-fabricated optima are interior for all five tasks and beat the squeezed tuned device on every task.\n* **Squeezing adds little to a converged re-fabricated device:** ≤ 0.3 % on all five tasks (≤ 0.3 % also with $\\Omega_a$, $g\\cosh r$, $\\epsilon_{\\rm eff}$ pinned). The 9–22 % of the previous version were gains over optima cut short by the box; the same effect is visible here as the 15 % Lorenz gain on the 300-evaluation ordinary optimum.\n* **Joint design:** at or above the ordinary optimum for four tasks; 7 % below it for channel equalization with weak squeezing, a gap that a local ordinary re-optimization from the joint device does not close. Lorenz with a relaxed photon budget (4, $N_c=16$): external squeezing adds about 4 % to a not fully converged device.\n* **Seeds:** tuned-drive gains 65 ± 3 % (channel equalization) and 26 ± 4 % (NARMA10) over eight fresh realizations.\n* **Emitter efficiency:** $\\eta_e=0.1$ costs a factor ≈ 4 in passes for a given precision; the channel-equalization gain at $10^7$ passes is 57 % instead of 63 %.\n* **Readout model:** the bin model is 7–13 % conservative where the input is vacuum; with squeezed input the vacuum-noise version underestimates the cavity noise by 24–58 % (the input-noise version is again conservative), but the measured gains change by about one percentage point because the emitter channel dominates.\n* **Convention fix:** $\\langle b_{\\rm in}^2\\rangle=-M$; only rotated-LO results of the squeezed-detection study change."),
]

def build(interp):
    n = nbf.v4.new_notebook(); n.cells = [nbf.v4.new_markdown_cell('# Second review round: boundary-free re-fabrication, seeds, emitter efficiency and a trajectory check of the readout'), nbf.v4.new_code_cell(HEADER)]
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == 'md' else nbf.v4.new_code_cell(src))
    fn = os.path.join(NB, '13_second_review_round.ipynb'); nbf.write(n, fn)
    return fn

if __name__ == '__main__':
    fn = build(None)
    if '--execute' in sys.argv:
        subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace', '--ExecutePreprocessor.timeout=1800', fn], check=True, cwd=NB)

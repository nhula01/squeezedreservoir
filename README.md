# squeezedreservoir

Code, data and notebooks for

> P. H. Nguyen and D. Soh, *Optical squeezing as a post-fabrication control of a cavity-QED reservoir computer* (2026).

A single cavity mode coupled to a two-level emitter is used as a quantum reservoir computer. The cavity is parametrically
driven (intracavity squeezing `r`) and coupled to a broadband squeezed bath (external squeezing `r_e`, relative phase
`dtheta`). These three pump-controlled parameters can be changed after the device is built. The code asks what they add to
a device whose drive has already been tuned, what an ordinary device with the same effective parameters or a fully
re-fabricated device achieves instead, and what training the controls costs on hardware.

## Layout

```
code/        simulator, run scripts, figure/macro scripts        (code/data -> ../data)
data/        every numerical array behind every figure, table and number (.npz)
notebooks/   one executed Jupyter notebook per study (problem statement, code, figures)
logs/        console logs of the long runs
```

Running `make_figs.py`, `make_numbers.py` and `make_rev.py` writes figures to `paper/figs/` and LaTeX macro/table files to
`paper/`; the directory is created on first use.

## Model and simulator

`code/sqz.py` integrates the Lindblad master equation

```
H0 = w a†a + wq σ†σ − (λ/2)(a² + a†²) + g(a†σ + aσ†),   H_in = i ε f(t) (e^{−iφ_d} a − e^{iφ_d} a†)
dρ/dt = −i[H0 + H_in, ρ] + D[L_a]ρ + γ D[σ]ρ,          L_a = √κ (cosh r_e a + e^{iθ} sinh r_e a†)
```

in the Bogoliubov frame of the parametric drive (the intracavity squeezing is exact; the Fock truncation `N_c` applies to
the quasi-mode). Superoperators act on the real vector space of Hermitian matrices, and the input-dependent propagator is
evaluated at `K = 7` Chebyshev nodes of the input and interpolated barycentrically. Features are
`Q, P, Q², P², σx, σy` at `N_v = 4` virtual times per symbol; the readout is ridge regression with
`δ = N_tr σ_rel²`, `σ_rel = 0.01`. `effective_parameters()` returns ε_eff, Ω_a, g cosh r, g sinh r, the emitter
relaxation rates and the stationary quadrature variances.

`code/expt.py` adds the experimental layer: impure squeezed bath (injection efficiency, phase jitter), homodyne readout
model with finite detection efficiencies, slow drift, and a loss evaluated as an experiment would measure it.

Units: rates and frequencies in units of the fabricated cavity detuning ω_a; time in 1/ω_a. Default device:
`ω_a = ω_q = 1, g = 0.5, κ = 0.2, γ = 0.1, ε = 0.2, T_in = 2, N_v = 4`.

## Studies and scripts

| Study | Scripts | Data | Notebook |
|---|---|---|---|
| Model, effective parameters, propagator accuracy | `sqz.py` | – | 00 |
| Mackey–Glass loss landscape (fixed drive) | `run_fig2.py` | `fig2_*.npz` | 01 |
| Optimizer comparison | `run_fig3.py`, `common.py` | `fig3_*.npz` | 02 |
| Feature sensitivity, task gradient, QFI | `run_fig4.py` | `fig4_lines.npz` | 03 |
| Four tasks, re-fabrication grid (fixed drive) | `run_tasks.py`, `run_lorenz_flip.py` | `tasks.npz`, `lorenz_flip.npz` | 04 |
| Robustness, software baselines, seeds | `run_supp.py`, `run_review.py` | `supp_*.npz`, `review.npz` | 05 |
| Santa Fe laser | `run_laser.py` | `laser.npz` | 06 |
| Spoken digits (FSDD) | `run_asr.py`, `asr_core.py`, `run_asr_base.py` | `asr*.npz` | 07, 12 |
| Base device first, convergent derivatives | `run_base.py`, `run_deriv.py` | `base.npz`, `deriv.npz` | 08 |
| Tuned drive, full re-fabrication, mapped device | `rev_base2.py`, `rev_refab.py`, `rev_post.py` | `rev_base2.npz`, `rev_refab.npz`, `rev_post.npz` | 09 |
| Measurement cost, gradients, measured-loss training | `rev_calib.py`, `rev_grad.py`, `rev_train*.py`, `rev_snr.py`, `rev_lo*.py` | `rev_calib.npz`, … | 10 |
| Non-ideal squeezing and drift | `rev_nonideal.py` | `rev_nonideal.npz` | 11 |
| Squeezing on the re-fabricated device (original box) | `rev_refab_sq.py`, `rev_refab_sq14.py` | `rev_refab_sq*.npz` | 13 |
| Wide-box re-fabrication with the noiseless objective (weak-signal optimum) | `rev_bounds.py` | `rev_bounds_<task>.npz` | 13 |
| Wide-box re-fabrication with a measurement-aware objective: squeezing on it, matched-parameter and mapped controls, joint search, photon budget | `rev_bounds2.py`, `rev_bounds2_final.py`, `rev_bounds2_joint.py`, `rev_bounds2_jointzero.py`, `rev_bounds2_jointlocal.py`, `rev_bounds2_budget.py` | `rev_bounds2_<task>*.npz` | 13 |
| Squeezing on re-fabricated devices for harder tasks (difficulty ladders) | `rev_hard.py`, `rev_hard_conv.py`, `rev_hard_mg20b.py` | `rev_hard_<task>*.npz` | 13 |
| Fresh input realizations of the tuned-drive gains | `rev_seeds.py` | `rev_seeds.npz` | 13 |
| Emitter detection efficiency | `rev_etae.py` | `rev_etae.npz` | 13 |
| Quantum-trajectory check of the readout model | `rev_sme.py` | `rev_sme.npz` | 13 |

`run_digits*.py` and `probe_digits*.py` are superseded two-speaker digit protocols kept for the record.

## Reproduce

```bash
pip install -r requirements.txt
export OMP_NUM_THREADS=1
cd code
# original studies (fixed drive)
python3 run_fig2.py; python3 run_fig3.py mg; python3 run_fig4.py
for p in fd fock noise narma ridge; do python3 run_supp.py $p; done
python3 run_tasks.py; python3 run_lorenz_flip.py
for p in gap baselines box seeds refab device2; do python3 run_review.py $p; done
python3 run_laser.py; python3 run_base.py; python3 run_deriv.py
python3 run_asr.py all; python3 run_asr_base.py            # needs the FSDD recordings, see below
# tuned drive, re-fabrication, experimental cost, non-idealities
for s in rev_calib rev_refab rev_base2 rev_post rev_grad rev_nonideal rev_train rev_snr rev_asr2 rev_train2 rev_lo rev_lo_diag rev_refab_sq rev_refab_sq14; do python3 $s.py; done
# second review round (about 6 h on 2 cores)
python3 rev_bounds.py narma nce
python3 rev_bounds2.py narma lorenz mg & python3 rev_bounds2.py laser nce; wait
python3 rev_bounds2_final.py mg narma lorenz nce laser; python3 rev_bounds2_jointzero.py mg narma lorenz nce laser
python3 rev_bounds2_jointlocal.py nce; python3 rev_bounds2_budget.py
python3 rev_seeds.py; python3 rev_etae.py; python3 rev_sme.py
python3 rev_hard.py lorenz5 lorenz10 nce6 & python3 rev_hard.py mg20 mg40 nce4; wait
python3 rev_hard_conv.py mg20 mg40; python3 rev_hard_mg20b.py
# figures, tables, macros
python3 make_figs.py; python3 make_numbers.py; python3 make_rev.py
# notebooks
python3 build_notebooks.py; python3 build_rev_notebooks.py; python3 build_rev2_notebook.py --execute
```

Every run script saves after each completed unit and resumes from its `.npz`, so it can be interrupted. One reservoir
evaluation (1,000 symbols, `N_c = 10`) takes about 1.4 s on one core; the full set of runs is roughly 20 core-hours, of
which the second-round re-fabrication study is about half.

**Spoken digits.** Download the [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset),
unzip it so that `free-spoken-digit-dataset-master/recordings` exists, and set `FSDD_DIR` to the parent directory
(default `~/fsdd`). The extracted features are cached in `data/asr_features.npz`, so the reservoir part of the digit
study runs without the audio. The Santa Fe laser series is included (`data/santafe_laser_A_10093.txt`).

## Requirements

Python ≥ 3.10, numpy, scipy, matplotlib; librosa and scikit-learn for spoken digits; nbformat, nbconvert and ipykernel
to rebuild the notebooks; `pdftoppm` (poppler) to display figures in the notebooks.

## Citation

If you use this code, please cite the paper above (see `CITATION.cff`).

## License

MIT (see `LICENSE`).

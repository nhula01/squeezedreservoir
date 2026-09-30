"""Revision 2: the study of rev_bounds.py with a measurement-aware objective, in an even wider box.

Why. With the noiseless objective of the paper (ridge delta = N_tr sigma_rel^2 on standardized features, i.e. every feature
measured to 1% of its own scale whatever its absolute size), widening the re-fabrication box moves the optimum to the
opposite faces: NARMA10 goes to the smallest drive of the box (eps = 0.01, max <a^dag a> = 1e-4) and channel
equalization to the smallest coupling (rev_bounds.py, ordinary stage). There the 1%-of-scale assumption is unphysical,
because the features are then far below the detector noise of a real readout. The objective below keeps the paper's ridge
for features measured to better than 1% and raises the ridge of every other feature to its actual readout-noise variance:
the readout model of Sec. 2.6 at N_rep = 1e7 passes per setting (the budget at which the paper's features reach 1%),
with the input-noise detector model validated by rev_sme.py (expt.ResLO, single_run_cov_sq) and the noise-matched ridge
expt.evaluate_nm. For every feature measured to 1% or better it is the paper's objective.

Box: g 0.005-3, kappa 0.005-3, gamma 0.002-2 (log), omega_a, omega_q 0-5 (linear; 0 is a symmetry point, not a bound,
since the unsqueezed dynamics with both detunings reversed is the complex conjugate one), eps 0.001-5 (log).
Budgets: ordinary 506 evaluations; sequential = ordinary after 300 + 206 squeezing evaluations = 506; joint 506.
Usage: python3 rev_bounds2.py narma lorenz ...   -> data/rev_bounds2_<task>.npz"""
import rev_bounds as rb
import rev_base2
from rev_common import *
NREP_OBJ = 1e7

def evaluate_meas(dev, th, task):
    """Measurement-aware loss: (L, held-out NRMSE, nmax)."""
    res = ResLO(dev, th); F, E, nm = res.run_moments(task.f)
    cov = single_run_cov_sq(dev, F, E, res.Vin); nv = noise_var_vector(dev, task, cov, NREP_OBJ)
    L, nr = evaluate_nm(dev, task, F, nv)
    return L, nr, nm

def evaluate_rel(dev, th, task):
    return evaluate(dev, th, task)

rb.evaluate = evaluate_meas; rev_base2.evaluate = evaluate_meas
rb.LOW[:] = [0.005, 0.005, 0.002, 0.0, 0.0, 0.001]; rb.HIW[:] = [3.0, 3.0, 2.0, 5.0, 5.0, 5.0]
rb.B_MAIN, rb.B_TOT, rb.N_LHS, rb.N_START = 300, 506, 150, 3

if __name__ == '__main__':
    import sys
    which = sys.argv[1:] or list(TASKS)
    rf = np.load(os.path.join(DATA, 'rev_refab.npz'), allow_pickle=True)
    rs = np.load(os.path.join(DATA, 'rev_refab_sq.npz'), allow_pickle=True)
    for name in which:
        task = TASKS[name]; fp = os.path.join(DATA, f'rev_bounds2_{name}.npz')
        out = dict(np.load(fp, allow_pickle=True)) if os.path.exists(fp) else {}
        save = lambda: np.savez(fp, **out)
        old = rf[f'{name}_best']; p_old = old[:6]; seed = 400 + list(TASKS).index(name)
        # reference points under both objectives: tuned base, tuned squeezed, old-box re-fabrication with and without squeezing
        if 'refs' not in out:
            b2, th2 = squeezed_point2(name); sq = rs[f'{name}_best']; dold = rb.device(p_old)
            pts = [(b2, np.zeros(3)), (b2, th2), (dold, np.zeros(3)), (replace(dold, phi_d=sq[0]), sq[1:])]
            out['refs'] = np.array([[*evaluate_meas(d, t, task), *evaluate_rel(d, t, task)] for d, t in pts])
            log(name, 'refs (meas L, rel L): tuned base %.4e %.4e | tuned sq %.4e %.4e | old refab %.4e %.4e | old refab+sq %.4e %.4e' % tuple(out['refs'][:, [0, 3]].ravel()))
            save()
        if 'ord_trace' not in out:
            obj = rb.Obj(task, rb.B_TOT, False); rb.global_search(obj, 6, [rb.to_u(np.clip(p_old, rb.LOW, rb.HIW))], seed, rb.B_MAIN, rb.B_TOT)
            out['ord_trace'] = np.array(obj.trace); save()
        T = out['ord_trace']; b300 = rb.best_row(T, rb.B_MAIN); bord = rb.best_row(T)
        out['ord_best'] = bord; out['ord_best500'] = b300; uo = rb.to_u(bord[:6]); out['ord_interior'] = np.array([min(x, 1 - x) for x in uo])
        log(name, 'ORDINARY', dict(zip(rb.PN, bord[:6].round(4))), 'L %.4e (300: %.4e) nmax %.3f faces' % (bord[10], b300[10], bord[12]),
            [rb.PN[k] for k in range(6) if out['ord_interior'][k] < 0.02])
        if 'seq_best' not in out:
            pd, th, L, S, n = rb.squeeze_on(rb.device(b300[:6]), task, b300[10])
            out['seq_best'] = np.array([*b300[:6], *th, pd, *L]); out['seq_scans'] = S; out['seq_evals'] = np.array(rb.B_MAIN + n)
            log(name, 'SEQUENTIAL', th.round(3), 'phi_d %.2f L %.4e -> %.4e (%.1f%%; vs ordinary %.1f%%) nmax %.2f' % (
                pd, b300[10], L[0], 100 * (1 - L[0] / b300[10]), 100 * (1 - L[0] / bord[10]), L[2])); save()
        if 'mat_best' not in out:
            pd, th, L, dvm, S, n = rb.matched_search(rb.device(bord[:6]), task)
            out['mat_best'] = np.array([dvm.g, dvm.kappa, dvm.gamma, dvm.omega, dvm.omega_q, dvm.eps, *th, pd, *L]); out['mat_scans'] = S
            log(name, 'MATCHED', th.round(3), 'phi_d %.2f L %.4e -> %.4e (%.1f%%)' % (pd, bord[10], L[0], 100 * (1 - L[0] / bord[10]))); save()
        if 'map_trace' not in out:
            sb = out['seq_best']; p0, Lmap, trm, e = rb.mapped_reopt(rb.device(sb[:6], sb[9]), sb[6:9], task)
            out['map_p0'] = p0; out['map_L0'] = Lmap; out['map_trace'] = trm; out['map_eff'] = np.array([[a, b] for a, b in e.items()], dtype=object)
            log(name, 'MAPPED %.4e, re-optimized %.4e (sequential %.4e)' % (Lmap[0], rb.best_row(trm)[10], sb[10])); save()
        jf = os.path.join(DATA, f'rev_bounds2_{name}_joint.npz')
        if 'joint_trace' not in out and os.path.exists(jf):          # joint stage run separately by rev_bounds2_joint.py
            out['joint_trace'] = np.load(jf)['joint_trace']; save()
        if 'joint_trace' not in out:
            obj = rb.Obj(task, rb.B_TOT, True); rb.global_search(obj, 10, [np.concatenate([rb.to_u(np.clip(p_old, rb.LOW, rb.HIW)), np.zeros(4)])], seed + 50, rb.B_MAIN, rb.B_TOT)
            out['joint_trace'] = np.array(obj.trace); save()
        bj = rb.best_row(out['joint_trace']); out['joint_best'] = bj
        log(name, 'JOINT', dict(zip(rb.PN, bj[:6].round(4))), 'sq', bj[6:10].round(3), 'L %.4e (vs ordinary %.1f%%) nmax %.2f' % (bj[10], 100 * (1 - bj[10] / bord[10]), bj[12]))
        if 'checks' not in out:
            sb, mb = out['seq_best'], out['mat_best']
            out['chk_ord'] = rb.check(rb.device(bord[:6]), (0, 0, 0), task, bord[12])
            out['chk_seq'] = rb.check(rb.device(sb[:6], sb[9]), sb[6:9], task, sb[12])
            out['chk_mat'] = rb.check(rb.device(mb[:6], mb[9]), mb[6:9], task, mb[12])
            out['chk_joint'] = rb.check(rb.device(bj[:6], bj[9]), bj[6:9], task, bj[12])
            out['rel'] = np.array([evaluate_rel(rb.device(bord[:6]), (0, 0, 0), task), evaluate_rel(rb.device(sb[:6], sb[9]), sb[6:9], task),
                                   evaluate_rel(rb.device(bj[:6], bj[9]), bj[6:9], task)])
            out['checks'] = np.array(1)
            log(name, 'Nc checks: ord', out['chk_ord'][:, 0], 'seq', out['chk_seq'][:, 0], 'mat', out['chk_mat'][:, 0], 'joint', out['chk_joint'][:, 0],
                '| relative objective at the optima (ord, seq, joint):', out['rel'][:, 0])
        save()
    log('done bounds2', which)

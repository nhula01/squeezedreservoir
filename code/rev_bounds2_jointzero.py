"""Revision 2 helper: the joint optimum of rev_bounds2.py evaluated with its squeezing switched off (same device, same drive),
to separate what the joint search gains through squeezing from what it gains through a better ordinary device.
Output: data/rev_bounds2_<task>_jointzero.npz  (L, NRMSE, nmax at zero squeezing; N_c = 10 and the checked cutoff)."""
import rev_bounds2 as r2
import rev_bounds as rb
from rev_common import *
if __name__ == '__main__':
    import sys
    for name in sys.argv[1:]:
        d = np.load(os.path.join(DATA, f'rev_bounds2_{name}.npz'), allow_pickle=True); j = rb.best_row(d['joint_trace']); task = TASKS[name]
        z = r2.evaluate_meas(rb.device(j[:6]), (0, 0, 0), task); c = rb.check(rb.device(j[:6]), (0, 0, 0), task, z[2])
        np.savez(os.path.join(DATA, f'rev_bounds2_{name}_jointzero.npz'), L=np.array(z), chk=c)
        log(name, 'joint optimum %.4e; same device unsqueezed %.4e (checked %.4e); squeezing share %.1f%%' % (j[10], z[0], c[-1][0], 100 * (1 - j[10] / z[0])))

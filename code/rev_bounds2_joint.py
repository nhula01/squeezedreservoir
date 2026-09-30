"""Revision 2 helper: the joint stage of rev_bounds2.py alone, written to data/rev_bounds2_<task>_joint.npz, so that it can run
in parallel with the other stages; rev_bounds2.py merges it (same seed and protocol as its own joint stage)."""
import rev_bounds2 as r2
import rev_bounds as rb
from rev_common import *

if __name__ == '__main__':
    import sys
    rf = np.load(os.path.join(DATA, 'rev_refab.npz'), allow_pickle=True)
    for name in sys.argv[1:]:
        task = TASKS[name]; p_old = rf[f'{name}_best'][:6]; seed = 400 + list(TASKS).index(name)
        obj = rb.Obj(task, rb.B_TOT, True)
        rb.global_search(obj, 10, [np.concatenate([rb.to_u(np.clip(p_old, rb.LOW, rb.HIW)), np.zeros(4)])], seed + 50, rb.B_MAIN, rb.B_TOT)
        np.savez(os.path.join(DATA, f'rev_bounds2_{name}_joint.npz'), joint_trace=np.array(obj.trace))
        log(name, 'joint-only done', rb.best_row(obj.trace)[10])

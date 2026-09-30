from run_digits import *
import sys
items = load_subset(); f, frames = build_sequence(items)
rng = np.random.default_rng(1); utts = np.arange(len(items)); digits = np.array([it[0] for it in items]); train_utts, test_utts = [], []
for d in range(10):
    u = utts[digits == d]; rng.shuffle(u); train_utts += list(u[:len(u) // 2]); test_utts += list(u[len(u) // 2:])
probes = {'flip_r3': (replace(DEV, s=-1.0), (0.3, 0, 0)), 'flip_r5': (replace(DEV, s=-1.0), (0.5, 0, 0)), 'flip_r5_re3': (replace(DEV, s=-1.0), (0.5, 0.3, 1.57)),
          'eps05': (replace(DEV, eps=0.5), (0, 0, 0)), 'eps10': (replace(DEV, eps=1.0), (0, 0, 0)), 'g09': (replace(DEV, g=0.9), (0, 0, 0)), 'Tin4': (replace(DEV, T_in=4.0), (0, 0, 0))}
for lab in sys.argv[1:]:
    dev, th = probes[lab]; r = digit_task(dev, th, items, frames, f, train_utts, test_utts); print(lab, 'L=%.5f train=%.3f test=%.3f nmax=%.2f' % r, flush=True)

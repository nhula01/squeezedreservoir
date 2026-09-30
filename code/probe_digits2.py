from run_digits import *
import sys
items = load_subset(); f, frames = build_sequence(items)
sp = np.array([it[1] for it in items]); utts = np.arange(len(items))
train_utts, test_utts = list(utts[sp == 'jackson']), list(utts[sp == 'nicolas'])
print('frontend', [frontend_baseline(items, frames, f, train_utts, test_utts, k) for k in (1, 3, 10)], flush=True)
probes = {'unsq': (DEV, (0, 0, 0)), 'mgopt': (DEV, (0.0, 0.2, -0.79)), 'nceopt': (DEV, (0.5, 0.5, -1.57)), 'flip_r5': (replace(DEV, s=-1.0), (0.5, 0, 0)), 'eps10': (replace(DEV, eps=1.0), (0, 0, 0))}
for lab in sys.argv[1:]:
    dev, th = probes[lab]; r = digit_task(dev, th, items, frames, f, train_utts, test_utts); print(lab, 'L=%.5f train=%.3f test=%.3f nmax=%.2f' % r, flush=True)

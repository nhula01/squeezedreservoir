"""Shared definitions for the referee-driven revision (fabrication baseline, experimental cost, non-idealities)."""
from common import *
from sqz import make_lorenz, make_nce, Task, effective_parameters
from expt import *
X_LASER = np.loadtxt(os.path.join(DATA, 'santafe_laser_A_10093.txt')).astype(float)
TASKS = {'mg': MG, 'narma': NARMA, 'lorenz': make_lorenz(horizon=2), 'nce': make_nce(),
         'laser': Task('laser', X_LASER[:1000], np.roll(X_LASER[:1000], -1), 100, 700, 1)}
NAMES = {'mg': 'Mackey-Glass', 'narma': 'NARMA10', 'lorenz': 'Lorenz-63', 'nce': 'channel eq.', 'laser': 'Santa Fe laser'}
NMAX = 2.5
BASE = np.load(os.path.join(DATA, 'base.npz'), allow_pickle=True)

def base_device(name, Nc=12):
    """Drive-optimized unsqueezed base device of Section 2.5 (eps*, phi_d of the squeezed optimum)."""
    return replace(DEV, eps=float(BASE[f'{name}_epsbest']), Nc=Nc)

def squeezed_point(name):
    return replace(base_device(name), phi_d=float(BASE[f'{name}_bestpd'])), np.array(BASE[f'{name}_best'], float)

# ---- revised base (drive amplitude AND drive frequency optimized at zero squeezing), from rev_base2.py
def _b2():
    p = os.path.join(DATA, 'rev_base2.npz')
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None

def base_device2(name, Nc=10):
    b2 = _b2(); eps2, dl2, pd = b2[f'{name}_best'][:3]
    return replace(DEV, eps=float(eps2), omega=1 + float(dl2), omega_q=1 + float(dl2), phi_d=float(pd), Nc=Nc)

def squeezed_point2(name, Nc=10):
    b2 = _b2(); return base_device2(name, Nc), np.array(b2[f'{name}_best'][3:], float)

def training_task():
    """Task for the cost/training study: the largest squeezing gain over the revised base among the real-signal-like tasks."""
    b2 = _b2(); gains = {}
    for n in ('lorenz', 'nce', 'laser'):
        if f'{n}_best' in b2: gains[n] = 1 - b2[f'{n}_Lsq2'][0][0] / b2[f'{n}_base2'][2]
    return max(gains, key=gains.get)

import numpy as np, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, '..', 'data'); PAPER = os.path.join(HERE, '..', 'paper'); FIG = os.path.join(PAPER, 'figs')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'axes.titlesize': 8,
                     'pdf.fonttype': 42, 'figure.dpi': 150})
def ld(name):
    p = os.path.join(DATA, name)
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None
def label(ax, s, dx=-0.18, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontweight='bold', fontsize=10, va='bottom')

# ---------------------------------------------------------------- Fig 1b: channels sketch
def fig1():
    fig, ax = plt.subplots(figsize=(7.0, 1.7)); ax.axis('off')
    def box(x, y, w, h, txt, fc):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec='k', lw=0.8))
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=7.5)
    box(0.02, 0.55, 0.16, 0.35, 'intracavity\nsqueezing $r$', '#fde0b0')
    box(0.02, 0.08, 0.16, 0.35, 'external squeezing\n$(r_{\\rm e},\\Delta\\theta)$', '#fde0b0')
    box(0.34, 0.55, 0.22, 0.35, 'linear cavity response\n$\\Omega_a=\\omega_a/\\cosh 2r$, $e^{\\pm r}$', '#cfe2f3')
    box(0.34, 0.08, 0.22, 0.35, 'emitter nonlinearity\nphase-sensitive relaxation', '#d9ead3')
    box(0.72, 0.30, 0.26, 0.42, 'measured features\n$\\langle Q\\rangle,\\langle P\\rangle,\\langle Q^2\\rangle,\\langle P^2\\rangle$\n$\\langle\\sigma_x\\rangle,\\langle\\sigma_y\\rangle$', '#eeeeee')
    ax.annotate('', xy=(0.34, 0.72), xytext=(0.18, 0.72), arrowprops=dict(arrowstyle='->', lw=1))
    ax.annotate('', xy=(0.34, 0.25), xytext=(0.18, 0.25), arrowprops=dict(arrowstyle='->', lw=1))
    ax.annotate('', xy=(0.34, 0.60), xytext=(0.18, 0.40), arrowprops=dict(arrowstyle='->', lw=1, ls='--', color='0.4'))
    ax.text(0.26, 0.47, 'mean field: none\n(Eq. 5)', fontsize=6.5, color='0.35', ha='center')
    ax.annotate('', xy=(0.72, 0.60), xytext=(0.56, 0.72), arrowprops=dict(arrowstyle='->', lw=1))
    ax.annotate('', xy=(0.72, 0.42), xytext=(0.56, 0.25), arrowprops=dict(arrowstyle='->', lw=1))
    ax.annotate('', xy=(0.45, 0.55), xytext=(0.45, 0.43), arrowprops=dict(arrowstyle='<->', lw=0.8, color='0.4'))
    ax.text(0.47, 0.47, '$g$', fontsize=7, color='0.35')
    ax.text(-0.005, 0.97, 'b', fontweight='bold', fontsize=10, transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(f'{FIG}/fig1_sketch.pdf', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- Fig 2: landscapes
def fig2():
    sc = ld('fig2_scans.npz'); pa = ld('fig2_paths.npz')
    if sc is None: return
    L0 = float(sc['L3'][0, 0, 0])
    rg, dg, reg = sc['rg'], sc['dg'], sc['reg']
    L1, L2, NR2 = sc['L1'], sc['L2'], sc['NR2']
    vmin = min(L1.min(), L2.min()); vmax = L0
    fig, axs = plt.subplots(2, 2, figsize=(6.0, 4.6), gridspec_kw=dict(wspace=0.55, hspace=0.45)); axs = axs.ravel()
    ax = axs[0]
    im = ax.pcolormesh(dg, rg, L1, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='viridis_r', shading='nearest')
    ax.set_xlabel(r'$\Delta\theta$ (rad)'); ax.set_ylabel('$r$'); ax.set_title(r'$\mathcal{L}(r,\Delta\theta)$ at $r_{\rm e}=%.2f$' % sc['re_fix'])
    ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    if 'T1' in sc:
        rc, dc, T1 = sc['rc'], sc['dc'], sc['T1']
        for i in range(len(rc)):
            for j in range(len(dc)):
                if T1[i, j] > 0.01:
                    ax.add_patch(plt.Rectangle((dc[j] - (dc[1] - dc[0]) / 2, rc[i] - (rc[1] - rc[0]) / 2), dc[1] - dc[0], rc[1] - rc[0],
                                               fill=False, hatch='////', ec='w', lw=0))
    if pa is not None:
        for p in pa['paths1']:
            p = np.asarray(p, float)
            ax.plot(p[:, 2], p[:, 0], '-o', color='w', ms=2, lw=0.8, mfc='w'); ax.plot(p[0, 2], p[0, 0], 's', color='w', ms=4); ax.plot(p[-1, 2], p[-1, 0], '*', color='w', ms=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04); cb.set_label(r'$\mathcal{L}$')
    ax = axs[1]
    im = ax.pcolormesh(reg, rg, L2, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='viridis_r', shading='nearest')
    ax.set_xlabel(r'$r_{\rm e}$'); ax.set_ylabel('$r$'); ax.set_title(r'$\mathcal{L}(r,r_{\rm e})$ at $\Delta\theta=%.2f$' % sc['dth_fix'])
    ax.plot(0, 0, 'x', color='r', ms=6, mew=1.5)
    if 'T2' in sc:
        rc, rec, T2 = sc['rc'], sc['rec'], sc['T2']
        for i in range(len(rc)):
            for j in range(len(rec)):
                if T2[i, j] > 0.01:
                    ax.add_patch(plt.Rectangle((rec[j] - (rec[1] - rec[0]) / 2, rc[i] - (rc[1] - rc[0]) / 2), rec[1] - rec[0], rc[1] - rc[0],
                                               fill=False, hatch='////', ec='w', lw=0))
    if pa is not None:
        for p in pa['paths2']:
            p = np.asarray(p, float)
            ax.plot(p[:, 1], p[:, 0], '-o', color='w', ms=2, lw=0.8, mfc='w'); ax.plot(p[0, 1], p[0, 0], 's', color='w', ms=4); ax.plot(p[-1, 1], p[-1, 0], '*', color='w', ms=7)
    ax.set_xlim(reg.min() - 0.01, reg.max() + 0.01); ax.set_ylim(rg.min() - 0.01, rg.max() + 0.01)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04); cb.set_label(r'$\mathcal{L}$')
    ax = axs[2]
    im2 = ax.pcolormesh(reg, rg, NR2, cmap='viridis_r', shading='nearest')
    ax.set_xlabel(r'$r_{\rm e}$'); ax.set_ylabel('$r$'); ax.set_title('held-out NRMSE')
    cb = fig.colorbar(im2, ax=ax, fraction=0.05, pad=0.04); cb.set_label('NRMSE')
    ax = axs[3]
    j0 = 0
    ax.plot(rg, L2[:, j0] / L0, 'k-', lw=1.2, label=r'$\mathcal{L}/\mathcal{L}_0$')
    ax.axhline(1, color='0.6', lw=0.6, ls='--'); ax.set_xlabel(r'$r$ (at $r_{\rm e}=0$)'); ax.set_ylabel(r'$\mathcal{L}/\mathcal{L}_0$')
    ax2 = ax.twinx(); ax2.plot(rg, sc['NM2'][:, j0], color='C1', lw=1); ax2.set_ylabel(r'max $\langle a^\dagger a\rangle$', color='C1')
    for a, s in zip(axs, 'abcd'): label(a, s)
    fig.savefig(f'{FIG}/fig2_landscape.pdf', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- Fig 3: optimizers
def fig3(tag='mg', fname='fig3_optim.pdf', L0=None, Lstar=None):
    d = ld(f'fig3_{tag}.npz')
    if d is None: return
    B = int(d['budget']); x = np.arange(1, B + 1)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.1), gridspec_kw=dict(width_ratios=[1.3, 0.8, 1.1], wspace=0.45))
    names = dict(gd='gradient descent', nm='Nelder--Mead'.replace('--', '–'), rs='random search'); cols = dict(gd='C3', nm='C0', rs='0.5')
    ax = axs[0]
    for m in ('gd', 'nm', 'rs'):
        c = d[f'curve_{m}']; med = np.median(c, 0); q1, q3 = np.percentile(c, [25, 75], 0)
        ax.plot(x, med, color=cols[m], lw=1.3, label=names[m]); ax.fill_between(x, q1, q3, color=cols[m], alpha=0.18, lw=0)
    if L0 is not None: ax.axhline(L0, color='k', ls='--', lw=0.7)
    if Lstar is not None: ax.axhline(1.05 * Lstar, color='k', ls=':', lw=0.7)
    ax.set_xlabel('reservoir evaluations'); ax.set_ylabel(r'best-so-far $\mathcal{L}$'); ax.set_yscale('log'); ax.legend(frameon=False, loc='upper right', fontsize=6)
    ax = axs[1]
    data = [d[f'curve_{m}'][:, -1] for m in ('gd', 'nm', 'rs')]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True, flierprops=dict(ms=2))
    for patch, m in zip(bp['boxes'], ('gd', 'nm', 'rs')): patch.set_facecolor(cols[m]); patch.set_alpha(0.5)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['GD', 'NM', 'RS']); ax.set_ylabel(r'final $\mathcal{L}$')
    if L0 is not None: ax.axhline(L0, color='k', ls='--', lw=0.7)
    ax.set_yscale('log')
    ax = axs[2]
    Ls = [np.asarray(p, float)[:, 3] for p in d['paths']]
    vmin, vmax = min(l.min() for l in Ls), max(l.max() for l in Ls)
    for p in d['paths']:
        p = np.asarray(p, float)
        seg = np.where(np.abs(np.diff(p[:, 2])) > np.pi)[0] + 1
        for q in np.split(np.arange(len(p)), seg):
            ax.plot(p[q, 2], p[q, 1], '-', color='0.7', lw=0.6, zorder=1)
        sc_ = ax.scatter(p[:, 2], p[:, 1], c=p[:, 3], cmap='viridis_r', norm=LogNorm(vmin, vmax), s=8, zorder=2)
        ax.plot(p[-1, 2], p[-1, 1], 'k*', ms=5, zorder=3)
    ax.set_xlabel(r'$\Delta\theta$ (rad)'); ax.set_ylabel(r'$r_{\rm e}$'); ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    fig.colorbar(sc_, ax=ax, fraction=0.05, pad=0.03, label=r'$\mathcal{L}$')
    for a, s in zip(axs, 'abc'): label(a, s)
    fig.savefig(f'{FIG}/{fname}', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- Fig 4: mechanism
def fig4(Lstar=None):
    d = ld('fig4_lines.npz')
    if d is None: return
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(4, 3, hspace=0.15, wspace=0.5, left=0.07, right=0.70)
    gsr = fig.add_gridspec(2, 1, hspace=0.5, left=0.80, right=0.99)
    names = ['r', 're', 'dth']; xl = ['$r$', r'$r_{\rm e}$', r'$\Delta\theta$ (rad)']
    ctrl = ['r', r'r_{\rm e}', r'\Delta\theta']; cols = ['C3', 'C0', 'C2']
    for c, name in enumerate(names):
        if f'line_{name}' not in d: continue
        A = d[f'line_{name}']; pts = d[f'pts_{name}']; xv = pts[:, c]
        axL = fig.add_subplot(gs[0, c]); axS = fig.add_subplot(gs[1, c], sharex=axL); axQ = fig.add_subplot(gs[2, c], sharex=axL); axG = fig.add_subplot(gs[3, c], sharex=axL)
        axL.plot(xv, A[:, 9] * 1e3, color='k', lw=1.2)
        for i in range(3):
            axS.plot(xv, A[:, i], color=cols[i], lw=1, label='$i=%s$' % ctrl[i])
            axQ.plot(xv, A[:, 6 + i], color=cols[i], lw=1)
            axG.plot(xv, A[:, 3 + i], color=cols[i], lw=1)
        for a in (axS, axQ, axG): a.set_yscale('log')
        if c == 0:
            axL.set_ylabel(r'$\mathcal{L}\times10^{3}$'); axS.set_ylabel('$S_i$'); axQ.set_ylabel('$F_{ii}$ per sample'); axG.set_ylabel('$G_i$')
            axS.legend(frameon=False, ncol=3, loc='lower center', fontsize=6, handlelength=1, columnspacing=0.8)
        axG.set_xlabel(xl[c])
        for a in (axL, axS, axQ): plt.setp(a.get_xticklabels(), visible=False)
        if c == 2:
            axG.set_xticks([-np.pi, 0, np.pi]); axG.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        label(axL, 'abc'[c], dx=-0.3, dy=1.08)
    # d: decomposition of S_i by observable class at the basin point (r line at r=0 == basin)
    axd = fig.add_subplot(gsr[0])
    A = d['line_r']; Sg = A[0, 12:21].reshape(3, 3)          # r=0, re=re_fix, dth=dth_fix
    frac = Sg ** 2 / (Sg ** 2).sum(1, keepdims=True)
    bottom = np.zeros(3); labs = ['$Q,P$', '$Q^2,P^2$', r'$\sigma_x,\sigma_y$']
    for gi, gc in enumerate(['#cfe2f3', '#6fa8dc', '#93c47d']):
        axd.bar([0, 1, 2], frac[:, gi], bottom=bottom, color=gc, ec='k', lw=0.4, label=labs[gi]); bottom += frac[:, gi]
    axd.set_xticks([0, 1, 2]); axd.set_xticklabels(['$S_r$', r'$S_{r_{\rm e}}$', r'$S_{\Delta\theta}$'])
    axd.set_ylabel('share of $S_i^2$'); axd.legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, handlelength=0.8, columnspacing=0.6)
    axd.set_ylim(0, 1.2); label(axd, 'd', dx=-0.35, dy=1.02)
    # e: final loss reached by 3D GD initialized along the lines
    ax = fig.add_subplot(gsr[1])
    for key, mk, lab, col in (('init_r', 'o', 'initial $r$ (at $r_{\\rm e}=%.2f$)' % d['re_fix'], 'C3'), ('init_re', 's', r'initial $r_{\rm e}$ (at $r=0$)', 'C0')):
        if key in d:
            F = d[key]; xv = F[:, 0] if key == 'init_r' else F[:, 1]
            ax.plot(xv, F[:, 3] * 1e3, mk, mfc='none', color=col, ms=4)
            ax.plot(xv, F[:, 4] * 1e3, mk, color=col, ms=4, label=lab)
    if Lstar is not None: ax.axhline(Lstar * 1e3, color='k', ls='--', lw=0.7)
    ax.set_xlabel(r'initial $r$ or $r_{\rm e}$'); ax.set_ylabel(r'$\mathcal{L}\times10^{3}$ (open: initial, filled: final)'); ax.legend(frameon=False, fontsize=6, loc='upper left', bbox_to_anchor=(0.0, 1.0)); ax.set_ylim(4.3, 8.2); label(ax, 'e', dx=-0.35, dy=1.02)
    fig.savefig(f'{FIG}/fig4_mech.pdf', bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- Supplement
def figS1():
    d = ld('supp_fd.npz')
    if d is None: return
    hs = d['fd_h']; fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.0), gridspec_kw=dict(wspace=0.4))
    for ax, name in zip(axs, ('unsqueezed', 'mid', 'basin')):
        G = d[f'fd_{name}']
        for i, (c, lab) in enumerate(zip(['C3', 'C0', 'C2'], ['$r$', r'$r_{\rm e}$', r'$\Delta\theta$'])):
            ax.plot(hs, G[:, i], 'o-', color=c, ms=3, lw=1, label=lab)
        ax.set_xscale('log'); ax.set_xlabel('finite-difference step $h$'); ax.set_title(name.replace('_', ' ') + r' $\vartheta=$(%.2f, %.2f, %.2f)' % tuple(d[f'anchor_{name}']), fontsize=7)
        ax.axhline(0, color='0.7', lw=0.5)
    axs[0].set_ylabel(r'$\partial\mathcal{L}/\partial\vartheta_i$'); axs[0].legend(frameon=False)
    fig.savefig(f'{FIG}/figS1_fd.pdf', bbox_inches='tight'); plt.close(fig)

def figS2(L0=None):
    d = ld('supp_noise.npz'); g = ld('supp_noisygd.npz')
    if d is None: return
    hs, sig = d['hs'], d['sig']
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.1), gridspec_kw=dict(wspace=0.45))
    for ax, name in zip(axs[:2], ('mid', 'basin')):
        G0, Gn = d[f'G0_{name}'], d[f'Gn_{name}']
        for i, (c, lab) in enumerate(zip(['C3', 'C0', 'C2'], ['$r$', r'$r_{\rm e}$', r'$\Delta\theta$'])):
            for a, h in enumerate(hs):
                frac = np.mean(np.sign(Gn[a, :, :, i]) == np.sign(G0[a, i]), axis=1)
                ax.plot(sig, frac, ['o', 's', '^'][a] + '-', color=c, ms=3, lw=0.8, label=f'{lab}, $h$={h:g}')
        ax.set_xscale('log'); ax.set_xlabel(r'measurement noise $\sigma_{\rm rel}$'); ax.set_ylabel('sign recovery'); ax.set_ylim(0.2, 1.05)
        ax.axhline(0.5, color='0.6', lw=0.6, ls='--'); ax.set_title(name + r' anchor $\vartheta=$(%.2f, %.2f, %.2f)' % tuple(d[f'anchor_{name}']), fontsize=7)
    axs[0].legend(frameon=False, fontsize=4.5, ncol=3, handlelength=1.2, loc='lower left')
    ax = axs[2]
    if g is not None:
        for key, c in (('curves_0.003', 'C0'), ('curves_0.01', 'C3')):
            if key in g:
                C = g[key]; ax.plot(np.arange(1, C.shape[1] + 1), np.median(C, 0), color=c, lw=1.2, label=r'noisy best-so-far, $\sigma_{\rm rel}$=' + key.split('_')[1])
                cl = g['clean_' + key.split('_')[1]]; ax.plot([C.shape[1]] * len(cl), cl, 'x', color=c, ms=5, label='noiseless loss at final setting')
        ax.plot([1] * len(g['L0_clean']), g['L0_clean'], 'k+', ms=5, label='initial (noiseless)')
    if L0 is not None: ax.axhline(L0, color='k', ls='--', lw=0.7)
    ax.set_xlabel('reservoir evaluations'); ax.set_ylabel(r'$\mathcal{L}$'); ax.legend(frameon=False, fontsize=4.5, loc='upper right')
    for a, s_ in zip(axs, 'abc'): label(a, s_)
    fig.savefig(f'{FIG}/figS2_noise.pdf', bbox_inches='tight'); plt.close(fig)

def figS3():
    d = ld('supp_narma_scan.npz'); o = ld('fig3_narma.npz')
    if d is None: return
    L3 = d['L3']; L0 = L3[0, 0, 0]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.0), gridspec_kw=dict(wspace=0.5))
    i, j, k = np.unravel_index(L3.argmin(), L3.shape)
    ax = axs[0]
    im = ax.pcolormesh(d['d3'], d['re3'], L3[i], norm=LogNorm(vmin=L3.min(), vmax=max(L0, L3[i].max())), cmap='viridis_r', shading='nearest')
    ax.set_xlabel(r'$\Delta\theta$ (rad)'); ax.set_ylabel(r'$r_{\rm e}$'); ax.set_title('NARMA10: $\\mathcal{L}(r_{\\rm e},\\Delta\\theta)$ at $r$=%.1f' % d['r3'][i]); fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax = axs[1]
    im = ax.pcolormesh(d['re3'], d['r3'], L3[:, :, k], norm=LogNorm(vmin=L3.min(), vmax=max(L0, L3[:, :, k].max())), cmap='viridis_r', shading='nearest')
    ax.plot(0, 0, 'rx', ms=6, mew=1.5); ax.set_xlabel(r'$r_{\rm e}$'); ax.set_ylabel('$r$'); ax.set_title(r'$\mathcal{L}(r,r_{\rm e})$ at $\Delta\theta$=%.2f' % d['d3'][k]); fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    ax = axs[2]
    if o is not None:
        B = int(o['budget']); x = np.arange(1, B + 1)
        for m, c, lab in (('gd', 'C3', 'gradient descent'), ('nm', 'C0', 'Nelder–Mead'), ('rs', '0.5', 'random search')):
            C = o[f'curve_{m}']; ax.plot(x, np.median(C, 0), color=c, lw=1.2, label=lab); ax.fill_between(x, *np.percentile(C, [25, 75], 0), color=c, alpha=0.18, lw=0)
        ax.axhline(L0, color='k', ls='--', lw=0.7); ax.set_yscale('log'); ax.legend(frameon=False); ax.set_xlabel('reservoir evaluations'); ax.set_ylabel(r'best-so-far $\mathcal{L}$')
    for a, s in zip(axs, 'abc'): label(a, s)
    fig.savefig(f'{FIG}/figS3_narma.pdf', bbox_inches='tight'); plt.close(fig)

def figS4():
    f = ld('supp_fock.npz'); r = ld('supp_ridge.npz')
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.0), gridspec_kw=dict(wspace=0.5))
    if f is not None:
        for key in f.files:
            if key.startswith('fock_'):
                A = f[key]; name = key[5:]
                axs[0].plot(A[:, 0], A[:, 1] / A[-1, 1], 'o-', ms=3, lw=1, label=name.replace('_', ' '))
                axs[1].plot(A[:, 0], A[:, 3], 'o-', ms=3, lw=1)
        axs[0].set_xlabel('Fock cutoff $N_c$'); axs[0].set_ylabel(r'$\mathcal{L}(N_c)/\mathcal{L}(20)$'); axs[0].legend(frameon=False, fontsize=6)
        axs[1].set_xlabel('Fock cutoff $N_c$'); axs[1].set_ylabel(r'max $\langle a^\dagger a\rangle$')
    if r is not None:
        L = r['L']; sigs = r['sigs']; ref = np.argmin(np.abs(sigs - 0.01))
        from scipy.stats import spearmanr
        ax = axs[2]
        for a, s in enumerate(sigs):
            rho = spearmanr(L[ref].ravel(), L[a].ravel()).correlation
            i, j, k = np.unravel_index(L[a].argmin(), L[a].shape)
            ax.plot(s, rho, 'ko', ms=4)
            ax.text(s, rho - 0.02, '(%.1f,%.1f,%.1f)' % (r['r3'][i], r['re3'][j], r['d3'][k]), fontsize=5, ha='center', va='top', rotation=0)
        ax.set_xscale('log'); ax.set_xlabel(r'assumed precision $\sigma_{\rm rel}$ (ridge)'); ax.set_ylabel('rank correlation with $\\sigma_{\\rm rel}$=1\\%'); ax.set_ylim(0.5, 1.05)
    for a, s in zip(axs, 'abc'): label(a, s)
    fig.savefig(f'{FIG}/figS4_fock_ridge.pdf', bbox_inches='tight'); plt.close(fig)

if __name__ == '__main__':
    sc = ld('fig2_scans.npz'); L0 = float(sc['L3'][0, 0, 0]) if sc is not None else None
    Lstar = None
    cands = []
    if sc is not None: cands += [sc['L3'].min(), sc['L1'].min(), sc['L2'].min()]
    f3 = ld('fig3_mg.npz')
    if f3 is not None: cands += [f3['final_gd'][:, 3].min(), f3['final_nm'][:, 3].min(), f3['final_rs'][:, 3].min()]
    if cands: Lstar = float(min(cands))
    fig1(); fig2(); fig3('mg', 'fig3_optim.pdf', L0, Lstar); fig4(Lstar); figS1(); figS2(L0); figS3(); figS4()
    print('figures done', L0, Lstar)

# ---------------------------------------------------------------- new Fig 1c and Fig 3 (tasks), effective-parameter table
TASKN = {'mg': 'Mackey–Glass', 'narma': 'NARMA10', 'lorenz': 'Lorenz-63', 'nce': 'channel eq.'}
TASKS_ = {'mg': 'MG', 'narma': 'NARMA', 'lorenz': 'Lorenz', 'nce': 'NCE'}
def fig1_effective():
    d = ld('tasks.npz')
    if d is None or 'effkeys' not in d: return
    keys = list(d['effkeys']); ix = {k: i for i, k in enumerate(keys)}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 1.9), gridspec_kw=dict(wspace=0.5))
    ax = axs[0]; A = d['effline_r']; x = d['effpts_r'][:, 0]
    ax.plot(x, A[:, ix['eps_eff']] / 0.2, label=r'$\epsilon_{\rm eff}/\epsilon$'); ax.plot(x, A[:, ix['Omega']], label=r'$\Omega_a/\omega_a$')
    ax.plot(x, A[:, ix['g_co']] / 0.5, label=r'$g\cosh r/g$'); ax.plot(x, A[:, ix['g_cr']] / 0.5, label=r'$g\sinh r/g$')
    ax.set_xlabel('$r$ (at $r_{\\rm e}=0$)'); ax.set_ylabel('linear parameters'); ax.legend(frameon=False, fontsize=6, ncol=2, loc='upper left', columnspacing=0.8, handlelength=1.2)
    for ax, nm, xl in ((axs[1], 're', r'$r_{\rm e}$ (at $r=0$, $\Delta\theta=-0.79$)'), (axs[2], 'dth', r'$\Delta\theta$ (at $r=0$, $r_{\rm e}=0.2$)')):
        A = d[f'effline_{nm}']; x = d[f'effpts_{nm}'][:, 1 if nm == 're' else 2]
        for k, c, lab in (('gamma_x', 'C3', r'$\gamma_x^{\rm eff}$'), ('gamma_y', 'C0', r'$\gamma_y^{\rm eff}$'), ('gamma_z', 'C2', r'$\gamma_z^{\rm eff}$')):
            ax.plot(x, A[:, ix[k]], color=c, lw=1, label=lab)
        rv = ld('review.npz')
        if rv is not None and f'gap_{nm}' in rv: ax.plot(x, rv[f'gap_{nm}'], color='k', lw=1, ls='-.', label='Liouvillian gap')
        ax.axhline(0.1, color='0.6', lw=0.6, ls=':'); ax.axhline(0.2, color='0.6', lw=0.6, ls='--')
        ax.set_ylabel('emitter rates'); ax.set_xlabel(xl)
        ax2 = ax.twinx(); ax2.plot(x, A[:, ix['VarQ']], color='0.4', lw=0.8, ls='-'); ax2.plot(x, A[:, ix['VarP']], color='0.4', lw=0.8, ls='--'); ax2.set_ylabel('Var$(Q)$ (—), Var$(P)$ (- -)', fontsize=6, color='0.4'); ax2.tick_params(labelsize=6, colors='0.4')
        if nm == 'dth': ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    axs[1].legend(frameon=False, fontsize=5.5, ncol=2, loc='upper left', columnspacing=0.8, handlelength=1.2)
    axs[0].text(-0.28, 1.05, 'c', transform=axs[0].transAxes, fontweight='bold', fontsize=10)
    fig.savefig(f'{FIG}/fig1_effective.pdf', bbox_inches='tight'); plt.close(fig)

def fig3_tasks():
    d = ld('tasks.npz')
    if d is None: return
    names = [n for n in ('mg', 'narma', 'lorenz', 'nce') if f'{n}_best' in d]
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.3), gridspec_kw=dict(wspace=0.55, width_ratios=[1.35, 1.1, 1.05, 0.9]))
    ax = axs[0]
    for i, n in enumerate(names):
        L0 = d[f'{n}_L3'][0, 0, 0]; Lb = float(d[f'{n}_bestL']); Lr = d[f'{n}_refab'].min()
        ax.bar(i - 0.18, Lb / L0, 0.34, color='C1', label='squeezed' if i == 0 else None)
        ax.bar(i + 0.18, Lr / L0, 0.34, color='0.6', label='best re-fabricated' if i == 0 else None)
        ax.text(i - 0.18, Lb / L0 + 0.02, '%.0f' % (100 * (1 - Lb / L0)), ha='center', fontsize=5.5, color='C1')
        ax.text(i + 0.18, Lr / L0 + 0.02, '%.0f' % (100 * (1 - Lr / L0)), ha='center', fontsize=5.5, color='0.3')
    ax.axhline(1, color='k', lw=0.7, ls='--'); ax.set_xticks(range(len(names))); ax.set_xticklabels([TASKS_[n] for n in names], fontsize=7, rotation=30, ha='right')
    ax.set_ylabel(r'$\mathcal{L}/\mathcal{L}_0$ (numbers: % reduction)'); ax.set_ylim(0, 1.12); ax.legend(frameon=False, fontsize=6, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, columnspacing=0.8, handlelength=1)
    ax = axs[1]
    for i, n in enumerate(names):
        C = d[f'{n}_gdcurves']; L0 = d[f'{n}_L3'][0, 0, 0]; x = np.arange(1, C.shape[1] + 1)
        ax.plot(x, np.median(C, 0) / L0, color=f'C{i}', lw=1.2, label=TASKN[n]); ax.fill_between(x, C.min(0) / L0, C.max(0) / L0, color=f'C{i}', alpha=0.15, lw=0)
        ax.axhline(d[f'{n}_L3'].min() / L0, color=f'C{i}', lw=0.6, ls='--')
    ax.set_xlabel('reservoir evaluations'); ax.set_ylabel(r'best-so-far $\mathcal{L}/\mathcal{L}_0$'); ax.legend(frameon=False, fontsize=5.5, loc='upper right')
    ax = axs[2]
    for i, n in enumerate(names):
        b = d[f'{n}_best']; L0 = d[f'{n}_L3'][0, 0, 0]; Lb = float(d[f'{n}_bestL'])
        ax.scatter(b[2], b[1], s=30 + 400 * (1 - Lb / L0), color=f'C{i}', alpha=0.8, edgecolor='k', lw=0.4)
        ax.annotate('%s\n$r$=%.2f' % (TASKS_[n], b[0]), (b[2], b[1]), textcoords='offset points', xytext={'nce': (-6, -30), 'narma': (12, 8), 'mg': (12, -6), 'lorenz': (10, -2)}[n], fontsize=5.5, ha='center' if n == 'nce' else 'left')
    ax.set_xlabel(r'$\Delta\theta$ (rad)'); ax.set_ylabel(r'$r_{\rm e}$'); ax.set_xlim(-np.pi - 0.3, np.pi + 0.3); ax.set_ylim(-0.03, 0.62); ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax = axs[3]
    for i, n in enumerate(names):
        N3 = d[f'{n}_N3']; nr0 = N3[0, 0, 0]
        # NRMSE at best setting: from scan if best came from scan else from gd
        L3 = d[f'{n}_L3']; gd = d[f'{n}_gd']
        if float(d[f'{n}_bestL']) <= L3.min() - 1e-15 and len(gd):
            nrb = gd[gd[:, 3].argmin(), 4]
        else:
            nrb = N3.ravel()[L3.argmin()]
        ax.plot(i, nr0, 'o', mfc='none', color=f'C{i}', ms=6); ax.plot(i, nrb, 'o', color=f'C{i}', ms=6)
        ax.plot([i, i], [nr0, nrb], color=f'C{i}', lw=0.8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([TASKS_[n] for n in names], fontsize=7, rotation=30, ha='right'); ax.set_ylabel('held-out NRMSE (open: unsqueezed)'); ax.set_xlim(-0.6, len(names) - 0.4)
    for a, s in zip(axs, 'abcd'): label(a, s, dx=-0.3)
    fig.savefig(f'{FIG}/fig3_tasks.pdf', bbox_inches='tight'); plt.close(fig)

def efftable():
    d = ld('tasks.npz')
    if d is None: return
    names = [n for n in ('mg', 'narma', 'lorenz', 'nce') if f'{n}_eff1' in d]
    rows = [('eps_eff', r'$\epsilon_{\rm eff}$'), ('Omega', r'$\Omega_a$'), ('g_co', r'$g\cosh r$'), ('g_cr', r'$g\sinh r$'),
            ('gamma_x', r'$\gamma_x^{\rm eff}$'), ('gamma_y', r'$\gamma_y^{\rm eff}$'), ('gamma_z', r'$\gamma_z^{\rm eff}$'),
            ('VarQ', r'$\mathrm{Var}(Q)$'), ('VarP', r'$\mathrm{Var}(P)$'), ('n_ss', r'$\langle a^\dagger a\rangle_{\rm ss}$'), ('ne_ss', r'$\langle\sigma^\dagger\sigma\rangle_{\rm ss}$')]
    from run_review import liouvillian_gap
    from common import DEV
    from sqz import replace
    gaps = {n: liouvillian_gap(replace(DEV, s=1.0), d[f'{n}_best']) for n in names}; gap0 = liouvillian_gap(DEV, (0, 0, 0))
    e0 = dict(d[f'{names[0]}_eff0'])
    out = ['\\begin{tabular}{l' + 'c' * (len(names) + 1) + '}', '\\toprule', 'Parameter & unsqueezed & ' + ' & '.join(TASKN[n] for n in names) + '\\\\', '\\midrule']
    out.append('$(r,\\re,\\dth)$ & $(0,0,0)$ & ' + ' & '.join('$(%.2f,%.2f,%.2f)$' % tuple(d[f'{n}_best']) for n in names) + '\\\\')
    for k, lab in rows:
        out.append(lab + ' & %.3f & ' % float(e0[k]) + ' & '.join('%.3f' % float(dict(d[f'{n}_eff1'])[k]) for n in names) + '\\\\')
    out.append('Liouvillian gap & %.3f & ' % gap0 + ' & '.join('%.3f' % gaps[n] for n in names) + '\\\\')
    out += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(PAPER, 'efftable.tex'), 'w').write('\n'.join(out))

def basetable():
    rv = ld('review.npz'); d = ld('tasks.npz'); nc = ld('nce_conv.npz'); lf = ld('lorenz_flip.npz')
    if rv is None or 'base_mg' not in rv: return
    names = ['mg', 'narma', 'lorenz', 'nce']
    rows = [('unsqueezed reservoir', lambda n: d[f'{n}_N3'][0, 0, 0]),
            ('squeezed reservoir', lambda n: {'mg': d['mg_gd'][d['mg_gd'][:, 3].argmin(), 4], 'narma': d['narma_gd'][d['narma_gd'][:, 3].argmin(), 4], 'lorenz': d['lorenz_N3'].ravel()[d['lorenz_L3'].argmin()], 'nce': nc['NR'][2]}[n]),
            ('squeezed, amplifying phase (Lorenz)', lambda n: lf['N3'].ravel()[lf['L3'].argmin()] if n == 'lorenz' else None),
            ('linear regression, 4 delays', lambda n: float(rv[f'base_{n}'][0][2])), ('linear regression, 10 delays', lambda n: float(rv[f'base_{n}'][1][2])),
            ('quadratic regression, 10 delays', lambda n: float(rv[f'base_{n}'][2][2])), ('100-node echo-state network (tuned)', lambda n: float(rv[f'base_{n}'][3][2]))]
    out = ['\\begin{tabular}{lcccc}', '\\toprule', 'Method & Mackey--Glass & NARMA10 & Lorenz-63 & channel eq.\\\\', '\\midrule']
    for lab, f in rows:
        vals = [f(n) for n in names]; out.append(lab + ' & ' + ' & '.join('--' if v is None else '%.3f' % v for v in vals) + chr(92) * 2)
    out += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(PAPER, 'basetable.tex'), 'w').write('\n'.join(out))

def figS_box():
    rv = ld('review.npz')
    if rv is None or 'box_nce' not in rv: return
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.4), gridspec_kw=dict(wspace=0.5)); g = rv['box_grid']
    for ax, nm, tt in zip(axs, ('nce', 'lorenzflip'), (r'channel eq., $\Delta\theta=-\pi/2$, $\phi_s=0$', r'Lorenz, $\Delta\theta=2.62$, $\phi_s=\pi$')):
        if f'box_{nm}' not in rv: continue
        L = rv[f'box_{nm}']; im = ax.pcolormesh(g, g, L, norm=LogNorm(), cmap='viridis_r', shading='nearest'); fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04, label=r'$\mathcal{L}$')
        NM = rv[f'boxNM_{nm}']
        for i in range(9):
            for j in range(9):
                if NM[i, j] > 8.0: ax.add_patch(plt.Rectangle((g[j] - 0.05, g[i] - 0.05), 0.1, 0.1, fill=False, hatch='////', ec='w', lw=0))
        b = rv[f'boxbest_{nm}']; ax.plot(b[1], b[0], 'r*', ms=8); ax.axhline(0.5, color='w', lw=0.6, ls='--'); ax.axvline(0.5, color='w', lw=0.6, ls='--')
        ax.set_xlabel('$r_{\\mathrm{e}}$'); ax.set_ylabel('$r$'); ax.set_title(tt, fontsize=7)
    for a, s_ in zip(axs, 'ab'): label(a, s_)
    fig.savefig(f'{FIG}/figS_box.pdf', bbox_inches='tight'); plt.close(fig)

def fig_real():
    ls_ = ld('laser.npz'); dg = ld('digits.npz')
    if ls_ is None or 'best' not in ls_: return
    from common import DEV, evaluate
    from run_laser import LASER
    from run_tasks import R3, RE3, D3
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.1), gridspec_kw=dict(wspace=0.6, width_ratios=[1.35, 0.9, 1.6, 1.0]))
    L3 = ls_['L3']; L0 = L3[0, 0, 0]; b = ls_['best']; k = np.argmin(np.abs(D3 - b[2]))
    ax = axs[0]; im = ax.pcolormesh(RE3, R3, L3[:, :, k], norm=LogNorm(), cmap='viridis_r', shading='nearest'); ax.plot(0, 0, 'rx', ms=6, mew=1.5); ax.plot(b[1], b[0], 'r*', ms=8); ax.set_xlabel(r'$r_{\mathrm{e}}$'); ax.set_ylabel('$r$'); ax.set_title(r'laser, $\Delta\theta=%.2f$; $\mathcal{L}$: %.0f (yellow) to %.0f' % (D3[k], L3[:, :, k].min(), L3[:, :, k].max()), fontsize=6.5)
    ax = axs[1]; C = ls_['gdcurves']; x = np.arange(1, C.shape[1] + 1)
    for c in C: ax.plot(x, c / L0, color='C1', lw=0.8, alpha=0.8)
    ax.axhline(L3.min() / L0, color='k', ls='--', lw=0.7); ax.set_xlabel('reservoir evaluations'); ax.set_ylabel(r'best-so-far $\mathcal{L}/\mathcal{L}_0$')
    ax = axs[2]; tr, te = LASER.split()
    _, _, _, w0, Z0 = evaluate(DEV, (0, 0, 0), LASER, return_all=True); _, _, _, w1, Z1 = evaluate(DEV, b, LASER, return_all=True)
    p0 = np.hstack([Z0, np.ones((len(Z0), 1))]) @ w0; p1 = np.hstack([Z1, np.ones((len(Z1), 1))]) @ w1
    idx = np.arange(te.start, te.start + 100); ax.plot(idx, LASER.y[idx], 'k-', lw=1, label='measured'); ax.plot(idx, p0[idx], color='0.55', lw=0.7, label='unsqueezed'); ax.plot(idx, p1[idx], color='C1', lw=0.7, label='squeezed')
    ax.set_xlabel('sample'); ax.set_ylabel('laser intensity'); ax.legend(frameon=False, fontsize=5.5, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.2)
    ax = axs[3]; asr = ld('asr.npz')
    if asr is not None and 'grid_m' in asr:
        G = asr['grid_m'][:, :, 0]; g = asr['grid']
        im = ax.pcolormesh(g, g, G, cmap='viridis_r', shading='nearest'); ax.plot(0, 0, 'rx', ms=6, mew=1.5)
        b = asr['best']; ax.plot(b[1], b[0], 'r*', ms=8); ax.set_xlabel(r'$r_{\mathrm{e}}$'); ax.set_ylabel('$r$')
        ax.set_title(r'spoken digits, $\phi_s=\pi$, $\Delta\theta=0$', fontsize=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label('training loss (fold 1)', fontsize=6); cb.ax.tick_params(labelsize=5)
    for a, s_ in zip(axs, 'abcd'): label(a, s_, dx=-0.3)
    fig.savefig(f'{FIG}/fig_real.pdf', bbox_inches='tight'); plt.close(fig)

TASKN5 = {'mg': 'MG', 'narma': 'NARMA10', 'lorenz': 'Lorenz', 'nce': 'NCE', 'laser': 'laser'}
def fig_base():
    b = ld('base.npz')
    if b is None: return
    names = [n for n in ('mg', 'narma', 'lorenz', 'nce', 'laser') if f'{n}_best' in b]
    if not names: return
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    for i, n in enumerate(names):
        E = b[f'{n}_eps']; L02 = E[np.isclose(E[:, 0], 0.2)][0, 1]; Lb = E[:, 1].min(); Ls = b[f'{n}_conv'][0, 0]
        ax.bar(i - 0.18, Lb / L02, 0.34, color='0.6', label='drive optimized' if i == 0 else None)
        ax.bar(i + 0.18, Ls / L02, 0.34, color='C1', label='drive + squeezing' if i == 0 else None)
        ax.text(i + 0.18, Ls / L02 * 1.15, '%.0f%%' % (100 * (1 - Ls / Lb)), ha='center', fontsize=6.5, color='C1')
    ax.axhline(1, color='k', ls='--', lw=0.7); ax.set_xticks(range(len(names))); ax.set_xticklabels([TASKN5[n] for n in names]); ax.set_ylabel(r'$\mathcal{L}/\mathcal{L}(\epsilon=0.2)$'); ax.set_yscale('log'); ax.set_ylim(0.01, 2.5)
    ax.legend(frameon=False, fontsize=6.5, loc='upper left', ncol=2)
    fig.savefig(f'{FIG}/fig_base.pdf', bbox_inches='tight'); plt.close(fig)

if __name__ == '__main__':
    fig_base(); fig_real(); fig1_effective(); fig3_tasks(); efftable(); basetable(); figS_box(); print('task figures done')

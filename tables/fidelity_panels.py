#!/usr/bin/env python3
# Fidelity tables (marginal panel, joint panel, AR ladder) from the result JSONs, and the numbers
# of the reference screen

import collections
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'paper')

COND = [('A', 'A'), ('F', 'B'), ('C', 'C'), ('DRH', 'D')]   # name in the files, name in the paper
BANDS = ('imd', 'oomd')
MAX_N = 6                                                   # channel counts with a NUTS reference
SCREEN = ('rhat_pred', 1.05)

MARG_DIR, JOINT_DIR, AR_DIR, DROP_FAILED_REF = 'marginal_c4_1000', 'joint_c4_1000', '', False
for i, a in enumerate(sys.argv):
    if a == '--marg_dir':
        MARG_DIR = sys.argv[i + 1]
    elif a == '--joint_dir':
        JOINT_DIR = sys.argv[i + 1]
    elif a == '--ar_dir':
        AR_DIR = sys.argv[i + 1]
    elif a == '--drop_failed_ref':
        DROP_FAILED_REF = True


def log(*a):
    print(*a, file=sys.stderr)


def results(folder):
    # every result file of the fidelity suite: (key, json), key = (band, N, draw, condition)
    for path in sorted(glob.glob(os.path.join(ROOT, folder, 'composed*.json'))):
        base = os.path.basename(path)[:-5]
        cond = re.search('_(DRH|A|F|C)(?:_[A-Za-z0-9.]+)?$', base)
        draw = re.search(r'_d(\d+)', base)
        n_match = re.search(r'N(\d+)', base)
        if not cond or not draw or not n_match:
            continue
        N = int(n_match.group(1))
        if N > MAX_N:
            continue
        band = 'imd' if '_imd' in base else 'oomd'
        yield (band, N, int(draw.group(1)), cond.group(1)), json.load(open(path))


def failed_refs(stat, tol):
    # configurations where the marginal or the joint NUTS reference has stat > tol
    out = set()
    for key, res in results(MARG_DIR):
        for entry in res.get('configs', []):
            nuts = ((entry.get('conditions') or {}).get(key[3]) or {}).get('nuts') or {}
            value = nuts.get(stat, {}).get('mean')
            if value is not None and float(value) > tol:
                out.add(key)
    for key, res in results(JOINT_DIR):
        values = [x for x in (res.get('per_row') or {}).get(stat) or [] if x is not None]
        if values and float(np.mean(values)) > tol:
            out.add(key)
    return out


def med(vals):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], float)
    return (float(np.median(a)), a.size) if a.size else (None, 0)


def fmt(x, p):
    return '--' if x is None else f'{x:.{p}f}'.replace('-', '$-$', 1)


def floor(m):
    # non-Gaussianity floor H(N) - H(q*) of one joint result
    parts = [m.get(x) for x in ('ppd_kl_total_self', 'ppd_kl_amort', 'ppd_decomp_residual')]
    return None if None in parts else parts[0] - parts[1] - parts[2]


def load_marginal(failed):
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for key, res in results(MARG_DIR):
        if key in failed:
            continue
        band, cond = key[0], key[3]
        for entry in res['configs']:
            methods = entry['conditions'].get(cond)
            if not methods or 'nuts' not in methods:
                continue
            get = lambda arm, k: (methods.get(arm) or {}).get(k, {}).get('mean')
            acc[band, cond]['kl_n_orc'].append(get('nuts', 'kl'))
            acc[band, cond]['kl_n_pfn'].append(get('pfn', 'kl_nuts_vs_pfn'))
            acc[band, cond]['kl_o_pfn'].append(get('pfn_m', 'kl_orc_vs_pfn'))
            acc[band, cond]['kl_o_n'].append(get('nuts', 'kl_orc_vs_nuts'))
    return acc


def load_joint(failed):
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for key, res in results(JOINT_DIR):
        m = res['mean']
        if key in failed or m.get('ppd_kl_amort') is None:
            continue
        a = acc[key[0], key[3]]
        a['diagnull'].append(m.get('ppd_kl_G_vs_diag'))
        a['kl_G_head'].append(m.get('ppd_kl_amort'))
        a['floor'].append(floor(m))
        a['kl_orc_h'].append(m.get('ppd_kl_orc_vs_h'))
        a['kl_o_n'].append(m.get('ppd_kl_orc_vs_nuts'))
    return acc


def load_ar(folder, failed):
    # joint NLL of oracle, our head, AR and our diagonal, per cell with each row's own K
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for key, res in results(folder):
        if key in failed:
            continue
        rows = res.get('per_row') or {}
        K = rows.get('K_cells') or []
        for f in ('nll_o', 'nll_h', 'nll_ar_tf', 'nll_diag'):
            values = rows.get(f) or []
            if K and len(values) == len(K):
                acc[key[0], key[3]][f].append(float(np.mean([a / b for a, b in zip(values, K) if b])))
    return acc


def captured(J):
    # 1 - median KL(N||Ours)/median KL(N||N_diag), in percent, and the median of the row ratios
    out = {}
    for band in BANDS:
        for raw, label in COND:
            r = J.get((band, raw), {})
            head = [x for x in r.get('kl_G_head', []) if x is not None]
            diag = [x for x in r.get('diagnull', []) if x is not None]
            if head and diag:
                ratios = [1 - a / b for a, b in zip(head, diag) if b]
                out[band, label] = (100 * (1 - np.median(head) / np.median(diag)), 100 * np.median(ratios), len(head))
    return out


def print_captured(tag, c):
    for band in BANDS:
        cells = [f'{lab} {c[band, lab][0]:>6.0f}% (row-med {c[band, lab][1]:>7.0f}%, n={c[band, lab][2]})'
                 for _, lab in COND if (band, lab) in c]
        if cells:
            log(f'[captured] {tag:11} {band:5} ' + ' | '.join(cells))


def screen_report(J, failed):
    # every number the screen paragraph quotes
    print_captured(f'screen {SCREEN[0]}>{SCREEN[1]}' if DROP_FAILED_REF else 'no screen', captured(J))
    if not DROP_FAILED_REF:
        return
    print_captured('no screen', captured(load_joint(set())))
    for stat, tol in (('rhat_pred', 1.01), ('rhat_pred', 1.05), ('rhat_pred', 1.1), ('chain_spread', 0.05)):
        f = failed_refs(stat, tol)
        c = captured(load_joint(f))
        bc = ' '.join(f"{p}={c['imd', p][0]:.0f}%" for p in ('B', 'C') if ('imd', p) in c)
        log(f'[captured] {stat}>{tol:<5} dropped={len(f):<4} in-prior {bc}')

    for band in BANDS:
        per = collections.Counter(k[2] for k in failed if k[0] == band)
        log(f'[screen] {band} dropped per draw {dict(sorted(per.items()))}, ' f'top five {sum(c for _, c in per.most_common(5))} of {sum(per.values())}')

    floor_dropped = []
    for band in BANDS:
        nuts_kl = {True: [], False: []}  # KL(oracle||NUTS), keyed by "dropped"
        ours_kl = {True: [], False: []}   # KL(oracle||Ours)
        pinned = [0, 0]
        for key, res in results(JOINT_DIR):
            if key[0] != band:
                continue
            m, dropped = res['mean'], key in failed
            if m.get('ppd_kl_orc_vs_nuts') is not None:
                nuts_kl[dropped].append(m['ppd_kl_orc_vs_nuts'])
            if m.get('ppd_kl_orc_vs_h') is not None:
                ours_kl[dropped].append(m['ppd_kl_orc_vs_h'])
            if dropped and floor(m) is not None:
                floor_dropped.append(floor(m))
            pinned[0] += m.get('n_pinned') or 0
            pinned[1] += (m.get('n_pinned') or 0) + (m.get('n_active') or 0)
        kept, drop = nuts_kl[False], nuts_kl[True]
        if kept and drop:
            log(f'[captured] {band} KL(oracle||NUTS) median: kept {np.median(kept):.4f} (n={len(kept)}), ' f'dropped {np.median(drop):.4f} (n={len(drop)}), ratio {np.median(drop) / np.median(kept):.1f}x')
        if ours_kl[False] and ours_kl[True]:
            log(f'[captured] {band} KL(oracle||Ours) median: kept {np.median(ours_kl[False]):.3f}, ' f'dropped {np.median(ours_kl[True]):.3f}')
        if pinned[1]:
            log(f'[screen] {band} kernel slots pinned to their true value: {100 * pinned[0] / pinned[1]:.1f}%')

    fl = np.array([x for r in J.values() for x in r.get('floor', []) if x is not None])
    if fl.size:
        q1, q3 = np.percentile(fl, [25, 75])
        log(f'[floor] retained n={fl.size}: median {np.median(fl):.2e}  IQR [{q1:.2e}, {q3:.2e}]  ' f'max {fl.max():.2e}  negative {int((fl < 0).sum())} (min {fl.min():.2e})')
    if floor_dropped:
        log(f'[floor] dropped n={len(floor_dropped)}: median {np.median(floor_dropped):.2e}')

    div, ess = {}, []
    for key, res in results(JOINT_DIR):
        if key[0] != 'imd' or key in failed:
            continue
        pr, nuts = res['per_row'], res['config']['nuts']
        div[key] = np.mean(pr['divergences']) / (nuts['chains'] * nuts['samples'])
        ess.append((np.mean(pr['ess_pred']), np.mean(pr['ess_lpd'])))
    d = np.array(list(div.values()))
    log(f'[screen] imd divergence rate: median {100 * np.median(d):.1f}%, 90th percentile {100 * np.percentile(d, 90):.1f}%')
    for thr in (0.005, 0.01, 0.02):
        c = captured(load_joint(failed | {k for k, v in div.items() if v > thr}))
        log(f"[captured] divergence rate>{thr:<5} in-prior B={c['imd', 'B'][0]:.0f}% C={c['imd', 'C'][0]:.0f}%")
    log(f'[screen] imd median ESS: predictive mean {np.median([e[0] for e in ess]):.0f}, log density {np.median([e[1] for e in ess]):.0f}')


def rows(acc, metrics, digits, bands=BANDS):
    # one LaTeX row per condition, the bands side by side
    out = []
    for raw, label in COND:
        cells = [fmt(med(acc.get((b, raw), {}).get(m, []))[0], digits) for b in bands for m in metrics]
        out.append(f'{label} & ' + ' & '.join(cells) + ' \\\\')
    return '\n'.join(out) + '\n'


def main():
    failed = failed_refs(*SCREEN) if DROP_FAILED_REF else set()
    if DROP_FAILED_REF:
        log(f'[screen] --drop_failed_ref: {len(failed)} configurations dropped, ' f'{SCREEN[0]} of the predictive mean > {SCREEN[1]}')
        totals = collections.Counter((k[0], k[3]) for k, _ in results(JOINT_DIR))
        for band in BANDS:
            k = sorted(x for x in failed if x[0] == band)
            if not k:
                continue
            log(f'  {band}: ' + ', '.join(f'N{n}/d{d}/{c}' for _, n, d, c in k))
            byc = collections.Counter(x[3] for x in k)
            byn = collections.Counter(x[1] for x in k)
            log(f'  {band} per condition: ' + ', '.join(f'{lab} {byc.get(raw, 0)}/{totals.get((band, raw), 0)}' for raw, lab in COND) + '   per N: ' + ', '.join(f'N{n} {byn[n]}' for n in sorted(byn)))

    marginal, joint = load_marginal(failed), load_joint(failed)
    log(f"[cells] marginal n={med(marginal['imd', 'A']['kl_n_orc'])[1]} configurations per row, " f"joint n={med(joint['imd', 'A']['kl_G_head'])[1]}")

    for band in BANDS:
        log(f'  band {band:7} marginal n per row {[med(marginal[band, raw]["kl_n_pfn"])[1] for raw, _ in COND]}' f'   joint n per row {[med(joint[band, raw]["kl_G_head"])[1] for raw, _ in COND]}')
    screen_report(joint, failed)

    tables = {'tab_fid_marg_panel.tex': rows(marginal, ('kl_o_n', 'kl_o_pfn', 'kl_n_pfn'), 3),
              'tab_fid_joint_panel.tex': rows(joint, ('kl_o_n', 'kl_orc_h', 'kl_G_head', 'diagnull'), 3),
              'tab_ar_ladder.tex': rows(load_ar(AR_DIR or JOINT_DIR, failed),
                                        ('nll_o', 'nll_h', 'nll_ar_tf', 'nll_diag'), 2, bands=('imd',))}
    
    os.makedirs(OUT, exist_ok=True)
    for name, body in tables.items():
        with open(os.path.join(OUT, name), 'w') as fh:
            fh.write(body + '\n')
        log(f'wrote paper/{name}')
    print('\n% ---- marginal panel ----\n' + tables['tab_fid_marg_panel.tex'])
    print('\n% ---- joint panel ----\n' + tables['tab_fid_joint_panel.tex'])


if __name__ == '__main__':
    main()

# tab_ablations.tex from a merged ablation json. Read-only; writes one .tex
import argparse, json
import numpy as np

# (arm name in the json, row label, group separator before this row)
ROWS = [
    ('JointTimePFN_l0.5_r32',        r'Full \method ($\lambda=0.5$, $R=32$)', False),
    ('JointTimePFN_worope_l0.5_r32',  'No RoPE',                              False),
    ('JointTimePFN_l0.5_r4',         r'$R=4$',                                True),
    ('JointTimePFN_l0.5_r8',         r'$R=8$',                                False),
    ('JointTimePFN_l0.5_r16',        r'$R=16$',                               False),
    ('JointTimePFN_l0.5_r32',        r'$R=32$',                               False),
    ('JointTimePFN_l0.5_r64',        r'$R=64$',                               False),
    ('JointTimePFN_l0.5_r128',       r'$R=128$',                              False),
    ('JointTimePFN_l0.0_r32',        r'$\lambda=0$',                          True),
    ('JointTimePFN_l0.25_r32',       r'$\lambda=0.25$',                       False),
    ('JointTimePFN_l0.5_r32',        r'$\lambda=0.5$',                        False),
    ('JointTimePFN_l0.75_r32',       r'$\lambda=0.75$',                       False),
    ('JointTimePFN_l1.0_r32',        r'$\lambda=1.0$',                        False),
    ('Marginal_JTPFN_nll',            'Marginal only, log score',             True),
    ('Marginal_JTPFN_beta',          r'Marginal only, $\beta$-NLL',           False),
]
COLS = [('mse', 4), ('crps_d', 4), ('nll', 3), ('jnll', 3), ('cov95', 3)]
CONDS = ('A', 'F', 'C', 'DRH')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--out', default='paper/tab_ablations.tex')
    p.add_argument('--ds', default='mimic_iv')
    a = p.parse_args()
    d = json.load(open(a.json))

    have = {k.split('|')[2] for k in d if len(k.split('|')) == 3}
    missing = sorted({r[0] for r in ROWS} - have)
    if missing:
        print(f'not in the json, rows will be dashed: {missing}')

    out, nwin = [], set()

    for arm, label, sep in ROWS:
        if sep:
            out.append(r'\midrule')
        cells = []
        for m, prec in COLS:
            vals, ns = [], []

            for c in CONDS:
                g = d.get(f'{a.ds}|{c}|{arm}', {})
                if m in g and g[m].get('mean') is not None and np.isfinite(g[m]['mean']):
                    vals.append(g[m]['mean']); ns.append(g[m]['n'])

            if len(vals) == len(CONDS):
                cells.append(f'{np.mean(vals):.{prec}f}')
                nwin.update(ns)
            else:
                cells.append('--')

        out.append(f'{label} & ' + ' & '.join(cells) + r' \\')

    with open(a.out, 'w') as fh:
        fh.write('\n'.join(out) + '\n')
    print(f'wrote {a.out}; window counts across cells: {sorted(nwin)}')


if __name__ == '__main__':
    main()

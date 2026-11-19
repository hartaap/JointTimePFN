#!/usr/bin/env python3
# The MIMIC-IV benchmark tables, from benchmark.merge's json
import argparse
import json
import sys

def num(x, prec):
    return f'{x:.{prec}f}'.replace('-', '$-$', 1)


# raw condition name in the results => the letter used in the paper
COND = [('A', 'A'), ('F', 'B'), ('C',  'C'), ('DRH', 'D')]

# (key in the json, row label)
ROWS = [
    ('timepfn_original', r'TimePFN'),
    ('tabpfn-ts', r'TabPFN-TS'),
    ('tabimpute', r'TabImpute'),
    ('chronos-2 (120M)', r'Chronos-2 (120M)'),
    ('chronos-2 (28M)', r'Chronos-2 (28M)'),
    ('tirex-2', r'TiRex-2'),
    (None, None),  # \midrule
    # the marginal row: same architecture and prior, marginal score only
    ('Marginal_JTPFN_nll', r'\method (marginal)'),
    ('JointTimePFN_l0.5_r32', r'\method (joint)'),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--metric', default='crps_d')
    p.add_argument('--metric2', default='',
                   help='second metric, a further block of condition columns after a spacer')
    p.add_argument('--dataset', default='mimic_iv')
    p.add_argument('--out', default='')
    p.add_argument('--joint', action='store_true',
                   help='the joint table (rows = condition; NLL_diag / NLL_joint / gain)')
    p.add_argument('--prec', type=int, default=4)
    p.add_argument('--ci', action='store_true',
                   help="append the bootstrap interval benchmark.merge stored beside each mean")
    p.add_argument('--bold', action='store_true', help='bold the lowest value in each column')
    a = p.parse_args()

    d = json.load(open(a.json))
    d.pop('_params', None)

    have = {k.split('|')[1] for k in d if '|' in k}
    gone = [raw for raw, _ in COND if raw not in have]
    if gone:
        print(f'% WARNING: the table wants conditions {gone} but the json holds {sorted(have)}; those columns are all dashes.', file=sys.stderr)

    # the modal group count, so a cell computed on fewer groups is marked
    nref = 0
    if a.ci:
        for key, row_label in ROWS:
            if key is None:
                continue
            for m in (a.metric, a.metric2):
                if not m:
                    continue
                for raw, paper_col in COND:
                    v = d.get(f'{a.dataset}|{raw}|{key}', {}).get(m)
                    if v is not None and v.get('n') is not None:
                        nref = max(nref, int(v['n']))

    def block(key, metric):
        out = []

        for raw, paper_col in COND:
            v = d.get(f'{a.dataset}|{raw}|{key}', {}).get(metric)

            if v is None:
                out.append('--')
                continue
            cell = num(v['mean'], a.prec)

            if a.ci:
                # the interval benchmark.merge stored
                if v.get('lo') is None or v.get('hi') is None:
                    print(f'% WARNING: no lo/hi for {key}|{raw}|{metric}; cell printed without an interval', file=sys.stderr)
                else:
                    cell += (f" {{\\scriptsize$[{v['lo']:.{a.prec}f},"
                             f"{v['hi']:.{a.prec}f}]$}}")
                if v.get('n') is not None and nref and int(v['n']) != nref:
                    cell += f" {{\\scriptsize$({int(v['n'])})$}}"
            out.append(cell)

        return out

    if a.joint:
        txt = joint_body(d, a.dataset, a.metric, prec=a.prec, ci=a.ci)
    else:
        body = []
        for key, label in ROWS:
            if key is None:
                body.append(r'\midrule')
                continue
            cells = block(key, a.metric)

            if a.metric2:
                cells += [''] + block(key, a.metric2)  # spacer column
                
            body.append((label, cells))

        if a.bold:
            # lowest value per column, among the rows that report it
            rows = [r for r in body if r != r'\midrule']
            for j in range(len(rows[0][1])):
                vals = [(float(c.replace('$-$', '-')), r) for r in rows for c in [r[1][j]] if c not in ('--', '')]
                if vals:
                    best = min(vals)[1]
                    best[1][j] = r'\textbf{' + best[1][j] + '}'
        txt = '\n'.join(r if r == r'\midrule' else f'{r[0]} & ' + ' & '.join(r[1]) + r' \\' for r in body) + '\n'

    if a.out:
        open(a.out, 'w').write(txt)
        print(f'wrote {a.out}', file=sys.stderr)
    print(txt)


def joint_body(d, dataset, key, prec=4, ci=False):
    # Joint table: diagonal NLL, joint NLL and gain per cond.
    out = []

    for raw, pap in COND:
        e = d.get(f'{dataset}|{raw}|{key}', {})
        c = []
        for m in ('jnll_diag', 'jnll', 'jgain'):
            v = e.get(m)
            if v is None:
                c.append('--')
                continue
            cell = num(v['mean'], prec)
            if ci and v.get('lo') is not None and v.get('hi') is not None:
                cell += f" {{\\scriptsize$[{v['lo']:.{prec}f},{v['hi']:.{prec}f}]$}}"
            c.append(cell)
        out.append(f'{pap} & ' + ' & '.join(c) + r' \\')
    return '\n'.join(out) + '\n'


if __name__ == '__main__':
    main()

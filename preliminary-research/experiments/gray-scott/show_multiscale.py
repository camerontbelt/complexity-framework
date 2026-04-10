import csv
from collections import defaultdict

rows = list(csv.DictReader(open(
    r'c:\Users\Cameron\Documents\complexity-framework\preliminary-research\gray_scott_multiscale.csv')))

profile = defaultdict(dict)
for r in rows:
    pf   = int(r['pool_factor'])
    name = r['name']
    if pf not in profile[name]:
        profile[name][pf] = []
    profile[name][pf].append(float(r['C_a']))

order = ['dead','static_spots','self_rep_spots','worm_complex','solitons','chaotic']
exp   = {'dead':'trivial','static_spots':'ordered','self_rep_spots':'complex',
         'worm_complex':'complex','solitons':'complex','chaotic':'chaotic'}
pfs   = [1, 2, 4, 8, 16]

print()
print('MULTI-SCALE C_a TABLE   (* = peak scale)')
hdr = '  '.join('x{:2d}'.format(p) for p in pfs)
print(f'  {"":18}  {"[class]":10}  {hdr}')
print('-' * 80)
for name in order:
    means = {pf: sum(profile[name][pf]) / len(profile[name][pf])
             for pf in pfs if pf in profile[name]}
    if not means:
        continue
    peak = max(means, key=means.get)
    vals = [('  {:.3f}{}'.format(means.get(pf, 0.0), '*' if pf == peak else ' '))
            for pf in pfs]
    print('  {:18s}  [{:8s}]  {}'.format(name, exp[name], '  '.join(vals)))

print()
print('Interpretation:')
print('  x1  = cell-level (128x128 = 16384 cells)')
print('  x2  = 64x64 = 4096 super-cells')
print('  x4  = 32x32 = 1024 super-cells')
print('  x8  = 16x16 =  256 super-cells  (entity scale used in earlier exp.)')
print('  x16 = 8x8   =   64 super-cells')

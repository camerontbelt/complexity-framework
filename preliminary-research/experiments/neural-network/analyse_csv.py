import csv, statistics

CSV = r'C:\Users\Cameron\Documents\complexity-framework\mnist_results.csv'
rows = []
with open(CSV, newline='') as f:
    for row in csv.DictReader(f):
        rows.append({k: float(v) if k not in ('model', 'seed') else v
                     for k, v in row.items()})

trained = [r for r in rows if r['model'] == 'trained'][0]
random  = [r for r in rows if r['model'] == 'random']

metrics = ['mean_H', 'std_H', 'op_up', 'op_down', 'mi1', 'decay',
           'tc_mean', 'gzip_ratio', 'wH', 'wOPs', 'wOPt', 'wT', 'wG', 'C']

hdr = ('metric', 'trained', 'rand_mean', 'rand_std', 'rand_min', 'rand_max', 'cohen_d')
print(f"{hdr[0]:>12}  {hdr[1]:>8}  {hdr[2]:>9}  {hdr[3]:>8}  {hdr[4]:>8}  {hdr[5]:>8}  {hdr[6]:>7}")
print('-' * 74)
for k in metrics:
    t  = trained[k]
    rv = [r[k] for r in random]
    mn = statistics.mean(rv)
    sd = statistics.stdev(rv)
    d  = (t - mn) / max(sd, 1e-9)
    print(f"{k:>12}  {t:>8.4f}  {mn:>9.4f}  {sd:>8.4f}  {min(rv):>8.4f}  {max(rv):>8.4f}  {d:>7.2f}")

# Percentile of trained in random distribution
print()
print("Percentile of trained model within random distribution:")
for k in metrics:
    t  = trained[k]
    rv = sorted(r[k] for r in random)
    pct = sum(1 for v in rv if v < t) / len(rv) * 100
    print(f"  {k:>12}: {pct:5.1f}th percentile")

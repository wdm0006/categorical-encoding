"""Benchmark for #239: ProcessPoolExecutor-parallel WOE fit/transform.

Measures wall-clock time of WOEEncoder.fit / .transform / .fit_transform for the
serial path (max_process=1) versus parallel paths (max_process>1) on synthetic
wide-category datasets, verifies output parity, and writes a JSON result block.

Run:  python3 benchmarks/bench-239-parallel-woe.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import statistics
import time

import numpy as np
import pandas as pd

from category_encoders import OrdinalEncoder, WOEEncoder

SHAPES = [
    # (name, rows, cols, cardinality)  -- cardinality = unique categories per column
    ('narrow-small (control)', 50_000, 5, 20),
    ('wide (representative)', 500_000, 20, 1_000),
    ('wide-tall', 1_000_000, 40, 100),
]
REPEATS = 3


def make_data(rows: int, cols: int, card: int, seed: int = 0):
    """Build a synthetic wide-category frame plus a binary target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f'c{i}': pd.Categorical(rng.integers(0, card, rows).astype(str)) for i in range(cols)})
    X.iloc[::1000, 0] = np.nan  # some missing values
    y = pd.Series(rng.integers(0, 2, rows), index=X.index)
    return X, y


def timed(fn, repeats: int) -> float:
    """Median wall-clock seconds of ``fn`` over ``repeats`` runs after a warmup."""
    fn()  # warmup
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def bench_shape(rows: int, cols: int, card: int, workers_list: list, repeats: int) -> dict:
    """Benchmark serial vs parallel WOE on one dataset shape."""
    X, y = make_data(rows, cols, card)
    X_t, _ = make_data(rows, cols, card, seed=1)  # transform input (fresh draws)

    entry = {'rows': rows, 'cols': cols, 'cardinality': card, 'runs': {}}

    # context: cost of the ordinal preprocessing WOE always pays serially
    ord_enc = OrdinalEncoder(cols=list(X.columns))
    t_ord_fit = timed(lambda: ord_enc.fit(X), repeats)
    t_ord_tr = timed(lambda: ord_enc.transform(X_t), repeats)
    entry['ordinal_context'] = {'fit_s': round(t_ord_fit, 4), 'transform_s': round(t_ord_tr, 4)}

    ref_out = None
    for workers in workers_list:
        run = {}
        enc = WOEEncoder(cols=list(X.columns), max_process=workers).fit(X, y)

        def do_fit():
            WOEEncoder(cols=list(X.columns), max_process=workers).fit(X, y)

        def do_transform():
            enc.transform(X_t)

        def do_fit_transform():
            WOEEncoder(cols=list(X.columns), max_process=workers).fit_transform(X, y)

        run['fit_s'] = timed(do_fit, repeats)
        run['transform_s'] = timed(do_transform, repeats)
        run['fit_transform_s'] = timed(do_fit_transform, repeats)

        # parity vs serial reference output
        out = enc.transform(X_t)
        if workers_list[0] == 1:
            ref_out = out
            run['parity'] = True
        else:
            run['parity'] = bool(out.equals(ref_out))

        entry['runs'][f'max_process={workers}'] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in run.items()}

    base = entry['runs'][f'max_process={workers_list[0]}']
    for _, run in entry['runs'].items():
        run['fit_speedup'] = round(base['fit_s'] / run['fit_s'], 2)
        run['transform_speedup'] = round(base['transform_s'] / run['transform_s'], 2)
        run['fit_transform_speedup'] = round(base['fit_transform_s'] / run['fit_transform_s'], 2)
    return entry


def main():
    """Run the benchmark matrix and write the JSON results + gate verdict."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='run the control + representative shapes only')
    args = ap.parse_args()

    shapes = SHAPES[:2] if args.quick else SHAPES
    results = {
        'environment': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'cpu_count': multiprocessing.cpu_count(),
            'machine': platform.machine(),
            'start_method': 'fork (process_creation_method default)',
        },
        'repeats': REPEATS,
        'shapes': [],
    }

    for shape_name, rows, cols, card in shapes:
        workers_list = [1, 2, 4, 8] if 'representative' in shape_name else [1, 4]
        print(f'=== {shape_name}: {rows} rows x {cols} cols x {card} cats, workers={workers_list} ===', flush=True)
        entry = bench_shape(rows, cols, card, workers_list, REPEATS)
        entry['name'] = shape_name
        results['shapes'].append(entry)
        for wname, run in entry['runs'].items():
            print(
                f"  {wname}: fit={run['fit_s']:>8.3f}s (x{run['fit_speedup']:<5}) "
                f"transform={run['transform_s']:>8.3f}s (x{run['transform_speedup']:<5}) "
                f"fit_transform={run['fit_transform_s']:>8.3f}s (x{run['fit_transform_speedup']:<5}) "
                f"parity={run['parity']}",
                flush=True,
            )

    # gate verdict: representative wide shape, best speedup vs serial
    rep = next(s for s in results['shapes'] if 'representative' in s['name'])
    parallel_runs = [run for wname, run in rep['runs'].items() if wname != 'max_process=1']
    best_ft = max(run['fit_transform_speedup'] for run in parallel_runs)
    best_fit = max(run['fit_speedup'] for run in parallel_runs)
    results['gate'] = {
        'representative_shape': rep['name'],
        'best_fit_transform_speedup': best_ft,
        'best_fit_speedup': best_fit,
        'threshold': 2.0,
        'passed_fit_transform': bool(best_ft >= 2.0),
        'passed_fit': bool(best_fit >= 2.0),
    }
    print(json.dumps(results['gate'], indent=2))
    with open('/home/user/work/benchmark-239-results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('full results written to /home/user/work/benchmark-239-results.json')


if __name__ == '__main__':
    main()

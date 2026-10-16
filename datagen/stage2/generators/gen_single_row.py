#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: LicenseRef-Axion-NC-1.0
# Licensed under the Axion Non-Commercial Research and Responsible Use
# License, Version 1.0. See the LICENSE file at the repository root.

"""Single-database unique identification: confirmed exoplanets.

Construction pattern (one of three used in Stage II):
    filter a candidate pool -> identify exactly one object by an observable
    property -> compute a derived quantity locally -> question and reference
    answer are produced together

Answers are computed at generation time; no LLM and no GPU are involved.
Question texts never name the source archive, so a solver has to work out
which catalogue holds confirmed exoplanet parameters.

A derived quantity is never the field used to single the planet out, and
several derived quantities are cross-checked against a tabulated column
(computed density against the catalogued density, for instance), so a
formula error shows up at generation time rather than in training data.

Data: NASA Exoplanet Archive, pscomppars table
    https://exoplanetarchive.ipac.caltech.edu/
    NASA material is generally not subject to copyright in the United States
    and may be used for any purpose, including commercially, provided NASA is
    acknowledged as the source and endorsement is not implied.

Usage:
    python gen_single_row.py --n 60 --out exo.jsonl
    python gen_single_row.py --n 60 --out exo.jsonl --cache ./data/pscomppars.csv
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
import random
import urllib.parse as P
import urllib.request as U

TAP = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync'
COLS = ('pl_name,hostname,sy_pnum,discoverymethod,disc_year,pl_orbper,pl_orbsmax,'
        'pl_rade,pl_bmasse,pl_dens,pl_orbeccen,pl_eqt,pl_insol,st_teff,st_rad,st_mass,sy_dist')

# Physical constants, in the units the catalogue uses.
G = 6.67430e-11            # m^3 kg^-1 s^-2
M_EARTH = 5.9722e24        # kg
R_EARTH = 6.371e6          # m
M_SUN = 1.98892e30         # kg
R_SUN = 6.957e8            # m
AU = 1.495978707e11        # m
SIGMA = 5.670374419e-8     # W m^-2 K^-4
L_SUN = 3.828e26           # W

CONST = {
    'G': 'G = 6.67430e-11 m^3 kg^-1 s^-2',
    'ME': 'M_Earth = 5.9722e24 kg',
    'RE': 'R_Earth = 6371 km',
    'MS': 'M_Sun = 1.98892e30 kg',
    'RS': 'R_Sun = 6.957e5 km',
    'AU': '1 au = 1.495978707e11 m',
    'SB': 'sigma = 5.670374419e-8 W m^-2 K^-4',
}

# Fields a planet can be singled out by: column -> (phrase, relative margin
# the winner must beat the runner-up by).
IDENT = {
    'pl_orbper':   ('orbital period', 0.05),
    'pl_rade':     ('planetary radius', 0.05),
    'pl_bmasse':   ('planetary mass', 0.05),
    'pl_dens':     ('bulk density', 0.05),
    'pl_orbeccen': ('orbital eccentricity', 0.10),
    'pl_eqt':      ('equilibrium temperature', 0.05),
    'pl_insol':    ('insolation flux', 0.08),
    'pl_orbsmax':  ('orbital semi-major axis', 0.05),
    'st_teff':     ('host star effective temperature', 0.03),
    'st_rad':      ('host star radius', 0.05),
    'st_mass':     ('host star mass', 0.05),
    'sy_dist':     ('distance from the Solar System', 0.03),
}


def surface_gravity(r):
    return G * r['pl_bmasse'] * M_EARTH / (r['pl_rade'] * R_EARTH) ** 2


def escape_velocity(r):
    return math.sqrt(2 * G * r['pl_bmasse'] * M_EARTH / (r['pl_rade'] * R_EARTH)) / 1000


def density(r):
    m = r['pl_bmasse'] * M_EARTH * 1000                      # g
    v = 4 / 3 * math.pi * (r['pl_rade'] * R_EARTH * 100) ** 3  # cm^3
    return m / v


def semi_major_axis(r):
    """Kepler's third law from the orbital period and the host star mass."""
    t = r['pl_orbper'] * 86400.0
    return (G * r['st_mass'] * M_SUN * t ** 2 / (4 * math.pi ** 2)) ** (1 / 3) / AU


def stellar_luminosity(r):
    return 4 * math.pi * (r['st_rad'] * R_SUN) ** 2 * SIGMA * r['st_teff'] ** 4 / L_SUN


def equilibrium_temperature(r):
    """Zero albedo, full redistribution."""
    return r['st_teff'] * math.sqrt(r['st_rad'] * R_SUN / (2 * r['pl_orbsmax'] * AU))


def transit_depth(r):
    return (r['pl_rade'] * R_EARTH / (r['st_rad'] * R_SUN)) ** 2 * 1e6   # ppm


def orbital_velocity(r):
    return 2 * math.pi * r['pl_orbsmax'] * AU / (r['pl_orbper'] * 86400.0) / 1000


def apoapsis(r):
    return r['pl_orbsmax'] * (1 + r['pl_orbeccen'])


def periapsis(r):
    return r['pl_orbsmax'] * (1 - r['pl_orbeccen'])


def hill_radius(r):
    m = r['pl_bmasse'] * M_EARTH / (r['st_mass'] * M_SUN)
    return r['pl_orbsmax'] * (1 - r['pl_orbeccen']) * (m / 3) ** (1 / 3) * AU / 1000


# Derived quantities: key -> (phrasing, function, unit, columns needed,
# constants to state in the question, tolerance, column to cross-check against)
DERIV = {
    'gravity':  ('the gravitational acceleration at its surface in m/s^2, treating it as a '
                 'uniform sphere', surface_gravity, 'm/s^2',
                 ('pl_bmasse', 'pl_rade'), ('G', 'ME', 'RE'), 0.01, None),
    'vesc':     ('its escape velocity in km/s, treating it as a uniform sphere',
                 escape_velocity, 'km/s',
                 ('pl_bmasse', 'pl_rade'), ('G', 'ME', 'RE'), 0.01, None),
    'density':  ('its mean density in g/cm^3, computed from its mass and radius',
                 density, 'g/cm^3',
                 ('pl_bmasse', 'pl_rade'), ('ME', 'RE'), 0.02, 'pl_dens'),
    'sma':      ('its orbital semi-major axis in au, from its orbital period and the mass of '
                 'its host star using Kepler\'s third law', semi_major_axis, 'au',
                 ('pl_orbper', 'st_mass'), ('G', 'MS', 'AU'), 0.02, 'pl_orbsmax'),
    'lum':      ('the bolometric luminosity of its host star in solar luminosities, from the '
                 'stellar radius and effective temperature', stellar_luminosity, 'L_Sun',
                 ('st_rad', 'st_teff'), ('RS', 'SB'), 0.03, None),
    'teq':      ('its equilibrium temperature in kelvin assuming zero albedo and full heat '
                 'redistribution', equilibrium_temperature, 'K',
                 ('st_teff', 'st_rad', 'pl_orbsmax'), ('RS', 'AU'), 0.02, 'pl_eqt'),
    'depth':    ('the depth in parts per million of the transit it would produce across its '
                 'host star, taking the ratio of the projected areas', transit_depth, 'ppm',
                 ('pl_rade', 'st_rad'), ('RE', 'RS'), 0.02, None),
    'vorb':     ('its mean orbital speed in km/s, taking the orbit as circular with the '
                 'tabulated semi-major axis', orbital_velocity, 'km/s',
                 ('pl_orbsmax', 'pl_orbper'), ('AU',), 0.01, None),
    'apo':      ('its apoapsis distance in au', apoapsis, 'au',
                 ('pl_orbsmax', 'pl_orbeccen'), (), 0.01, None),
    'peri':     ('its periapsis distance in au', periapsis, 'au',
                 ('pl_orbsmax', 'pl_orbeccen'), (), 0.01, None),
    'hill':     ('the radius in kilometres of its Hill sphere at periapsis',
                 hill_radius, 'km',
                 ('pl_bmasse', 'st_mass', 'pl_orbsmax', 'pl_orbeccen'), ('ME', 'MS', 'AU'),
                 0.02, None),
}

# A derived quantity must not restate the field the planet was identified by.
SAME = {
    'pl_dens': ('density',), 'pl_orbsmax': ('sma', 'apo', 'peri'),
    'pl_eqt': ('teq',), 'pl_orbper': ('sma', 'vorb'),
    'pl_orbeccen': ('apo', 'peri'),
}

NUM = ('sy_pnum', 'disc_year', 'pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_bmasse', 'pl_dens',
       'pl_orbeccen', 'pl_eqt', 'pl_insol', 'st_teff', 'st_rad', 'st_mass', 'sy_dist')


def load(cache):
    """Read the planetary-systems table, downloading it once if needed."""
    if cache and os.path.exists(cache):
        text = open(cache, encoding='utf-8').read()
    else:
        url = f'{TAP}?query=' + P.quote(f'select {COLS} from pscomppars') + '&format=csv'
        text = U.urlopen(U.Request(url, headers={'User-Agent': 'axion/1.0'}),
                         timeout=300).read().decode()
        if cache:
            os.makedirs(os.path.dirname(os.path.abspath(cache)) or '.', exist_ok=True)
            open(cache, 'w', encoding='utf-8').write(text)
    rows = []
    for d in csv.DictReader(io.StringIO(text)):
        for k in NUM:
            v = (d.get(k) or '').strip()
            d[k] = float(v) if v else None
        if d['pl_name']:
            rows.append(d)
    if not rows:
        raise SystemExit('no rows returned from the archive')
    return rows


def sig(x, n=3):
    if x == 0:
        return '0'
    return f'%.{n}g' % x


def make_pool(rows, rng):
    """Constrain the full table down to a pool, and describe the constraint."""
    kind = rng.choice(['method', 'year', 'dist', 'multi', 'teff'])
    if kind == 'method':
        m = rng.choice(sorted({r['discoverymethod'] for r in rows if r['discoverymethod']}))
        return [r for r in rows if r['discoverymethod'] == m], \
               f'confirmed exoplanets discovered by {m.lower()}'
    if kind == 'year':
        y = rng.randrange(2010, 2024)
        return [r for r in rows if r['disc_year'] == y], \
               f'confirmed exoplanets announced in {y}'
    if kind == 'dist':
        lo = rng.choice([10, 20, 30, 50, 80])
        hi = lo + rng.choice([10, 20, 30])
        return [r for r in rows if r['sy_dist'] and lo < r['sy_dist'] < hi], \
               f'confirmed exoplanets whose system lies between {lo} and {hi} parsecs from the Sun'
    if kind == 'multi':
        n = rng.choice([3, 4, 5, 6])
        return [r for r in rows if r['sy_pnum'] == n], \
               f'confirmed exoplanets in systems with exactly {n} known planets'
    lo = rng.choice([3000, 4000, 5000, 6000])
    hi = lo + 500
    return [r for r in rows if r['st_teff'] and lo < r['st_teff'] < hi], \
           f'confirmed exoplanets whose host star has an effective temperature between {lo} and {hi} K'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='output jsonl (appended, so runs can be resumed)')
    ap.add_argument('--n', type=int, default=60, help='number of questions to generate')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--cache', default='', help='cache the downloaded table at this path')
    ap.add_argument('--min-pool', type=int, default=8,
                    help='minimum planets in a pool before it is usable')
    ap.add_argument('--max-per-planet', type=int, default=2,
                    help='how many questions may share one planet')
    ap.add_argument('--max-tries', type=int, default=20000)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = load(args.cache)
    print(f'[data] {len(rows)} planets', flush=True)

    out, seen = [], set()
    if os.path.exists(args.out):
        for line in open(args.out):
            d = json.loads(line)
            out.append(d)
            seen.add(d['meta']['key'])
        print(f'[resume] {len(out)} questions already present', flush=True)

    per_planet = collections.Counter(d['meta']['planet'] for d in out)
    stat, tries = collections.Counter(), 0
    fo = open(args.out, 'a')

    while len(out) < args.n and tries < args.max_tries:
        tries += 1
        pool, pool_desc = make_pool(rows, rng)
        if len(pool) < args.min_pool:
            stat['pool_small'] += 1
            continue

        field, (fdesc, min_margin) = rng.choice(list(IDENT.items()))
        cand = [r for r in pool if r[field] is not None]
        if len(cand) < 4 or len(cand) < 0.8 * len(pool):
            stat['field_sparse'] += 1
            continue

        direction = rng.choice(['largest', 'smallest'])
        cand.sort(key=lambda r: r[field], reverse=(direction == 'largest'))
        top, runner = cand[0], cand[1]
        denom = max(abs(top[field]), abs(runner[field]), 1e-12)
        margin = abs(top[field] - runner[field]) / denom
        if margin < min_margin:
            stat['margin'] += 1
            continue

        dk = rng.choice(list(DERIV))
        if dk in SAME.get(field, ()):
            stat['same_qty'] += 1
            continue
        ddesc, fn, unit, need, consts, tol, check = DERIV[dk]
        if any(top[c] is None for c in need):
            stat['deriv_missing'] += 1
            continue
        if dk in ('apo', 'peri', 'hill') and not top['pl_orbeccen']:
            stat['zero_ecc'] += 1
            continue

        try:
            val = fn(top)
        except Exception:
            stat['calc_err'] += 1
            continue
        if val is None or not math.isfinite(val) or val == 0:
            stat['bad_val'] += 1
            continue
        # Cross-check against the catalogued value where the archive has one.
        if check and top[check] is not None and top[check] != 0:
            if abs(val - top[check]) / abs(top[check]) > 0.25:
                stat['crosscheck'] += 1
                continue

        key = f'{top["pl_name"]}|{dk}'
        if key in seen:
            stat['dup'] += 1
            continue
        if per_planet[top['pl_name']] >= args.max_per_planet:
            stat['planet_cap'] += 1
            continue

        cstr = '; '.join(CONST[c] for c in consts)
        question = (f'Consider all {pool_desc}. Among them, take the one with the {direction} '
                    f'{fdesc}. For that planet, determine {ddesc}.'
                    + (f' Use {cstr}.' if cstr else '')
                    + ' Give the numerical value to three significant figures.')
        row = {
            'id': 'exo-' + hashlib.md5(key.encode()).hexdigest()[:8],
            'walk': 'EXO',
            'question': question,
            'ground_truth': f'{sig(val)} {unit}',
            'answer_type': 'numeric',
            'tolerance': tol,
            'tree': {'id': 'root', 'type': 'sequential', 'children': [
                {'id': 'identify', 'critical': False, 'expected': top['pl_name'],
                 'desc': f'Identify the planet: {direction} {fdesc} in the pool'},
                {'id': 'params', 'critical': False,
                 'expected': {c: top[c] for c in need},
                 'desc': 'Retrieve the required parameters'},
                {'id': 'final', 'critical': True, 'expected': sig(val),
                 'desc': f'{dk} value'}]},
            'meta': {'key': key, 'planet': top['pl_name'], 'host': top['hostname'],
                     'pool': pool_desc, 'n_pool': len(cand), 'id_field': field,
                     'direction': direction, 'id_value': top[field],
                     'runner_up': runner['pl_name'], 'runner_value': runner[field],
                     'margin': round(margin, 4), 'deriv': dk,
                     'inputs': {c: top[c] for c in need}, 'value': val},
        }
        out.append(row)
        seen.add(key)
        per_planet[top['pl_name']] += 1
        stat['ok'] += 1
        fo.write(json.dumps(row, ensure_ascii=False) + '\n')
        fo.flush()
        print(f'{len(out):4d}  {row["id"]}  {top["pl_name"]:<22} {field:<12} {direction:<8} '
              f'{dk:<8} -> {sig(val)} {unit}', flush=True)

    print(f'DONE  {len(out)} questions  rejections: {dict(stat)}')


if __name__ == '__main__':
    main()

# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: LicenseRef-Axion-NC-1.0
# Licensed under the Axion Non-Commercial Research and Responsible Use
# License, Version 1.0. See the LICENSE file at the repository root.

"""Leaderboard aggregation: constrain a pool, then take an extremum or a count.

Construction pattern (one of three used in Stage II). Questions have the shape

    "Among all <objects> satisfying <constraint> in <region>, which one has the
     largest <field>?"

The difficulty is not hop count but that no aggregated page exists: no single
web page lists, say, every airport in one region ranked by elevation, so a
solver has to assemble the pool itself before comparing.

Multiple unrelated sources are used on purpose. Drawing every question from one
database teaches a model to map "this kind of question" to "that one site",
which is the path dependence this task is meant to avoid; it would also
exercise a single search strategy.

Answer shapes
    extremum      name of the extremal record (high answer entropy, and the
                  closest match to how such questions are really asked)
    count_thresh  how many pool members pass a threshold
    join_count    cross-source join: the pool comes from source A while the
                  deciding attribute must be looked up per item in GeoNames

Sources
    AIR   OurAirports              Public Domain
    GEO   GeoNames                 CC BY 4.0
    LANG  Glottolog                CC BY 4.0
    PWR   WRI Global Power Plant   CC BY 4.0
    OA    OpenAlex institutions

Usage
    python gen_ranked_list.py --src AIR --n 200 --out air.jsonl --cache ./data
    python gen_ranked_list.py --src AIR,GEO,LANG,PWR --n 800 --out rank.jsonl --cache ./data
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import urllib.request

_ap = argparse.ArgumentParser(
    description='Leaderboard-aggregation question generator (threshold filter + extremum/count).',
    epilog='example: python gen_ranked_list.py --src AIR --n 200 --out air.jsonl --cache ./data')
_ap.add_argument('--src', default='AIR',
                 help='source code(s), comma-separated: AIR, GEO, LANG, PWR, OA')
_ap.add_argument('--out', required=True, help='output jsonl (appended, so runs can be resumed)')
_ap.add_argument('--n', type=int, default=40, help='number of questions to generate')
_ap.add_argument('--seed', type=int, default=1)
_ap.add_argument('--cache', default='./data', help='cache directory for the raw datasets; missing files are downloaded')
_ap.add_argument('--pool-lo', type=int, default=20, help='minimum pool size')
_ap.add_argument('--pool-hi', type=int, default=60, help='maximum pool size')
_ap.add_argument('--sourceless', type=int, default=0,
                 help='1 = question text names no data source at all')
_args = _ap.parse_args()

SCR = _args.cache
SRC = os.path.join(SCR, 'ourairports.csv')
URL = 'https://davidmegginson.github.io/ourairports-data/airports.csv'
N = _args.n
SEED = _args.seed
OUT = _args.out
POOL_LO = _args.pool_lo
POOL_HI = _args.pool_hi
VER = 'rank'
rng = random.Random(SEED)

os.makedirs(SCR, exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(OUT)) or '.', exist_ok=True)
AS_OF = __import__('datetime').date.today().isoformat()


BLIND_SHORT = os.environ.get('BLIND_SHORT', '0') == '1'

def _fetch(path, url, binary=False):
    if not os.path.exists(path):
        print(f'[rank] downloading {url.split("?")[0][:60]} -> {path}', flush=True)
        urllib.request.urlretrieve(url, path)
    return path


def src_lang():
    """Glottolog languages, from the CLDF languages.csv.

        Carries latitude/longitude, so it supports the same spatial joins as the
        other geographic sources.
    """
    import csv as _c
    f = _fetch(f'{SCR}/glottolog_languages.csv',
               'https://raw.githubusercontent.com/glottolog/glottolog-cldf/master/cldf/languages.csv')
    rows = []
    for r in _c.DictReader(open(f, encoding='utf-8')):
        rows.append({'name': r.get('Name'), 'fam': r.get('Family_ID'),
                     'area': r.get('Macroarea'), 'ctry': r.get('Countries'),
                     'latitude': r.get('Latitude'), 'longitude': r.get('Longitude'),
                     'level': r.get('Level'), 'iso': r.get('ISO639P3code')})
    FAM = {}
    for r in rows:
        if r['fam'] and r['name']:
            FAM.setdefault(r['fam'], r['fam'])
    byfam, byarea = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        if r['fam']:
            byfam[r['fam']].append(r)
        if r['area'] and r['ctry']:
            byarea[(r['area'], (r['ctry'] or '').split(';')[0])].append(r)
    def _fam(k):
        nm = glotto_name(k) if SOURCELESS else None
        return f'languages of the {nm} family' if nm else \
               f'languages assigned to the family with identifier {k}'
    pools = [(v, _fam(k)) for k, v in byfam.items()]
    _fa = collections.defaultdict(list)
    for r in rows:
        if r['fam'] and r['area']:
            _fa[(r['fam'], r['area'])].append(r)
    for (f_, a_), v in _fa.items():
        _d = _fam(f_)
        if _d:
            pools.append((v, f'{_d} that are classified in the {a_} macroarea'))
    pools += [(v, f'languages of the {k[0]} macroarea that are recorded as spoken in '
                  + (country_name(k[1]) if SOURCELESS else f'country {k[1]}'))
              for k, v in byarea.items()]
    aggs = [('lat_gt', .40, 'latitude', 'how many of them have a recorded reference latitude '
                            'north of {t} degrees', None, 'recorded reference latitude'),
            ('lat_lt', .30, 'latitude', 'how many of them have a recorded reference latitude '
                            'south of {t} degrees', None, 'recorded reference latitude'),
            ('lon_gt', .30, 'longitude', 'how many of them have a recorded reference longitude '
                            'east of {t} degrees', None, 'recorded reference longitude')]
    blind = ('the standard comprehensive openly licensed catalogue of the world\'s languages '
             '(the one that assigns every language a stable alphanumeric code and records its '
             'family, macroarea, countries of use and a reference latitude/longitude)')
    if BLIND_SHORT: blind = "the standard open catalogue of the world's languages"
    return pools, aggs, blind


def src_oa():
    """OpenAlex institutions, enumerated per country. Thresholds on works_count / cited_by_count."""
    import urllib.request as _u
    CC = os.environ.get('OA_COUNTRIES',
        'is,ie,hr,si,sk,lt,lv,ee,lu,cy,mt,al,mk,me,ba,rs,md,ge,am,az,uy,py,bo,ec,cr,pa,'
        'gt,hn,ni,sv,jm,tt,tn,ma,sn,gh,ke,tz,ug,zm,zw,bw,na,mu,np,lk,kh,la,mn,kg,tj,uz,'
        'jo,lb,om,qa,kw,bh,nz,fj,pg').split(',')
    TYPE = {'education': 'degree-granting educational institutions',
            'healthcare': 'healthcare institutions', 'company': 'companies',
            'government': 'government bodies', 'nonprofit': 'non-profit organisations',
            'facility': 'research facilities', 'archive': 'archives'}
    pools = []
    for c in CC:
        f = f'{SCR}/openalex_inst_{c}.json'
        try:
            if not os.path.exists(f):
                req = _u.Request(f'https://api.openalex.org/institutions?'
                                 f'filter=country_code:{c}&per-page=200',
                                 headers={'User-Agent': 'axion/1.0 (mailto:noreply@example.com)'})
                with _u.urlopen(req, timeout=40) as rr, open(f, 'wb') as w:
                    w.write(rr.read())
            d = json.load(open(f))
        except Exception as e:
            print(f'[oa] {c} skipped: {type(e).__name__}', flush=True)
            continue
        by = collections.defaultdict(list)
        for x in d.get('results', []):
            t = x.get('type')
            if t in TYPE:
                by[t].append({'name': x.get('display_name'),
                              'works_count': x.get('works_count'),
                              'cited_by_count': x.get('cited_by_count')})
        for t, v in by.items():
            if SOURCELESS: continue
            pools.append((v, f'{TYPE[t]} based in {c.upper()}'))
    aggs = [('works_gt', .50, 'works_count', 'how many of them are credited with more than '
                              '{t} indexed works', 'number of indexed works'),
            ('cited_gt', .50, 'cited_by_count', 'how many of them have accumulated more than '
                              '{t} citations across their indexed works', 'total citation count')]
    blind = ('the large fully open bibliographic catalogue of scholarly works and the '
             'institutions credited on them (the one that gives every institution a stable '
             'identifier together with its country, type, indexed-work count and total '
             'citation count)')
    if BLIND_SHORT: blind = 'the large fully open bibliographic catalogue of scholarly works and their institutions'
    return pools, aggs, blind

RFC_ALL = []
_CITIES = None
def cities():
    """GeoNames cities15000: 34,152 cities above 15,000 inhabitants.

        Population and coordinates are always present. Used as the second hop of a
        cross-source join, where the pool comes from one source and the deciding
        attribute is looked up here per item.
    """
    global _CITIES
    if _CITIES is None:
        import zipfile
        f = _fetch(f'{SCR}/cities15000.zip',
                   'https://download.geonames.org/export/dump/cities15000.zip')
        z = zipfile.ZipFile(f)
        _CITIES = []
        for l in z.read(z.namelist()[0]).decode('utf-8', 'ignore').splitlines():
            q = l.split('\t')
            if len(q) >= 15 and q[14].isdigit():
                _CITIES.append({'name': q[1], 'ascii': q[2], 'cc': q[8],
                                'lat': float(q[4]), 'lon': float(q[5]),
                                'pop': int(q[14])})
    return _CITIES


def _haversine(a1, o1, a2, o2):
    from math import asin, cos, radians, sin, sqrt
    a1, o1, a2, o2 = map(radians, (a1, o1, a2, o2))
    return 2 * 6371 * asin(sqrt(sin((a2 - a1) / 2) ** 2 +
                                cos(a1) * cos(a2) * sin((o2 - o1) / 2) ** 2))


def nice(x):
    """Round a threshold to a human-looking value (nearest 50/100/500/1000)."""
    ax = abs(x)
    for u in (1000, 500, 100, 50, 10, 5, 1):
        if ax >= u * 4:
            return int(round(x / u) * u)
    return int(round(x))


def fnum(v):
    try:
        return float(v)
    except Exception:
        return None



def src_air():
    import csv as _c
    f = _fetch(f'{SCR}/ourairports.csv',
               'https://davidmegginson.github.io/ourairports-data/airports.csv')
    rows = list(_c.DictReader(open(f, encoding='utf-8')))
    TYPE = {'small_airport': 'small airports', 'medium_airport': 'medium-sized airports',
            'large_airport': 'large airports', 'heliport': 'heliports',
            'seaplane_base': 'seaplane bases', 'balloonport': 'balloonports'}
    CT = {}
    try:
        import pycountry
        for r in rows:
            k = r['iso_country']
            if k and k not in CT:
                c = pycountry.countries.get(alpha_2=k)
                CT[k] = c.name if c else k
    except Exception:
        for r in rows:
            CT.setdefault(r['iso_country'], r['iso_country'])
    byct, byrg = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        if r['type'] not in TYPE:
            continue
        byct[(r['iso_country'], r['type'])].append(r)
        if r['iso_region']:
            byrg[(r['iso_region'], r['type'])].append(r)
    pools = []
    if SOURCELESS:
        bysel = collections.defaultdict(list)
        for r in rows:
            if r['type'] not in TYPE:
                continue
            cc = r['iso_country']
            if (r.get('iata_code') or '').strip():
                bysel[(cc, 'iata')].append(r)
            if (r.get('scheduled_service') or '').strip().lower() in ('yes', '1', 'true'):
                bysel[(cc, 'sched')].append(r)
        SEL = {'iata': 'airports with an assigned IATA three-letter code',
               'sched': 'airports with scheduled commercial passenger service'}
        for (c, k), v in bysel.items():
            pools.append((v, f'{SEL[k]} in {country_name(c)}'))
        _a1 = {}
        try:
            _a1f = _fetch(f'{SCR}/admin1CodesASCII.txt',
                          'https://download.geonames.org/export/dump/admin1CodesASCII.txt')
            for _l in open(_a1f, encoding='utf-8'):
                _pp = _l.rstrip('\n').split('\t')
                if len(_pp) >= 2:
                    _a1[_pp[0]] = _pp[1]
        except Exception:
            _a1 = {}
        if _a1:
            byreg2 = collections.defaultdict(list)
            for r in rows:
                if r['type'] not in TYPE:
                    continue
                _rg = (r.get('iso_region') or '').replace('-', '.', 1)
                if _rg not in _a1:
                    continue
                if (r.get('iata_code') or '').strip():
                    byreg2[(_rg, 'iata')].append(r)
                if (r.get('scheduled_service') or '').strip().lower() in ('yes', '1', 'true'):
                    byreg2[(_rg, 'sched')].append(r)
            for (_rg, k), v in byreg2.items():
                pools.append((v, f'{SEL[k]} in {_a1[_rg]}, {country_name(_rg[:2])}'))
    else:
        for (c, t), v in byct.items():
            pools.append((v, f'{TYPE[t]} in {CT.get(c, c)}'))
        for (g, t), v in byrg.items():
            cc = v[0]['iso_country']
            pools.append((v, f'{TYPE[t]} in the {g} administrative region of {CT.get(cc, cc)}'))
    aggs = [('elev_gt', .35, 'elevation_ft', 'how many of them have a published field '
                             'elevation greater than {t} feet', None, 'published field elevation'),
            ('elev_lt', .20, 'elevation_ft', 'how many of them have a published field '
                             'elevation below {t} feet', None, 'published field elevation'),
            ('sched',   .25, None, 'how many of them are flagged as having scheduled '
                                   'commercial service'),
            ('iata',    .20, None, 'how many of them have an assigned three-letter IATA code')]
    blind = ('the standard openly published global aviation facility dataset (the one that '
             'assigns each facility a unique four-character identifier and records its type, '
             'country, field elevation, scheduled-service flag and IATA code)')
    if BLIND_SHORT: blind = 'the standard openly published global dataset of aviation facilities'
    return pools, aggs, blind


def src_pwr():
    """WRI Global Power Plant database, by country and fuel type.

        capacity_mw is populated for every record and is an officially published
        figure, so it agrees across independent sources and yields a unique answer
        even when the question names no database.
    """
    import csv as _c
    f = _fetch(f'{SCR}/gppd.csv',
               'https://raw.githubusercontent.com/wri/global-power-plant-database/'
               'master/output_database/global_power_plant_database.csv')
    rows = list(_c.DictReader(open(f, encoding='utf-8')))
    FUEL = {'Hydro': 'hydroelectric power stations', 'Solar': 'solar power stations',
            'Wind': 'wind power stations', 'Gas': 'natural-gas-fired power stations',
            'Coal': 'coal-fired power stations', 'Oil': 'oil-fired power stations',
            'Nuclear': 'nuclear power stations', 'Geothermal': 'geothermal power stations',
            'Biomass': 'biomass-fired power stations', 'Waste': 'waste-to-energy plants'}
    by = collections.defaultdict(list)
    for r in rows:
        fu = (r.get('primary_fuel') or '').strip()
        cn = (r.get('country_long') or '').strip()
        nm = (r.get('name') or '').strip()
        if fu not in FUEL or not cn or not nm:
            continue
        by[(cn, fu)].append(r)
    pools = []
    for (cn, fu), v in by.items():
        pools.append((v, f'{FUEL[fu]} in {cn}'))
    aggs = [('cap_gt',  .34, 'capacity_mw', 'how many of them have an installed capacity '
                             'greater than {t} MW', None, 'installed capacity in megawatts'),
            ('cap_lt',  .22, 'capacity_mw', 'how many of them have an installed capacity '
                             'below {t} MW', None, 'installed capacity in megawatts'),
            ('cyear_lt', .22, 'commissioning_year', 'how many of them were commissioned '
                              'before {t}', None, 'year of commissioning'),
            ('cyear_gt', .22, 'commissioning_year', 'how many of them were commissioned '
                              'after {t}', None, 'year of commissioning')]
    blind = ('the standard openly published global inventory of electricity generating '
             'facilities (the one recording each station\'s country, primary fuel, '
             'installed capacity in megawatts, commissioning year and owner)')
    if BLIND_SHORT: blind = 'the standard openly published global power plant inventory'
    return pools, aggs, blind


def src_geo():
    """GeoNames per-country gazetteers, one zip per country.

        Uses the dem column (SRTM elevation), which is populated for every record.
        The elevation and population columns are far too sparse to threshold on.
    """
    import zipfile
    CC = os.environ.get('GEO_COUNTRIES',
        'IS,IE,HR,SI,SK,LT,LV,EE,LU,CY,MT,AL,MK,ME,BA,RS,MD,GE,AM,AZ,'
        'UY,PY,BO,EC,CR,PA,GT,HN,NI,SV,JM,TT,'
        'TN,MA,SN,GH,KE,TZ,UG,ZM,ZW,BW,NA,MU,'
        'NP,LK,KH,LA,MN,KG,TJ,UZ,AM,JO,LB,OM,QA,KW,BH,'
        'NZ,FJ,PG,SB,VU,WS,TO').split(',')
    FCL = {'T': 'named mountains, hills and other terrain features',
           'H': 'named rivers, lakes and other hydrographic features',
           'P': 'populated places', 'S': 'named spots, buildings and farms',
           'A': 'administrative divisions', 'L': 'named parks and areas',
           'V': 'named forests and vegetation features'}
    CT = {}
    try:
        import pycountry
        for c in CC:
            o = pycountry.countries.get(alpha_2=c)
            CT[c] = o.name if o else c
    except Exception:
        CT = {c: c for c in CC}
    pools = []
    for c in sorted(set(CC)):
        f = f'{SCR}/geonames_{c}.zip'
        try:
            _fetch(f, f'https://download.geonames.org/export/dump/{c}.zip')
            z = zipfile.ZipFile(f)
            nm = [x for x in z.namelist() if x != 'readme.txt'][0]
            rows = []
            for l in z.read(nm).decode('utf-8', 'ignore').splitlines():
                p_ = l.split('\t')
                if len(p_) >= 17:
                    rows.append({'name': p_[1], 'fclass': p_[6], 'fcode': p_[7],
                                 'admin1': p_[10], 'dem': p_[16]})
        except Exception as e:
            print(f'[geo] {c} skipped: {type(e).__name__}', flush=True)
            continue
        by = collections.defaultdict(list)
        for r in rows:
            if r['fclass'] in FCL and r['admin1']:
                by[(r['fclass'], r['admin1'])].append(r)
        for (fc, a1), v in by.items():
            if fc == 'A':
                continue
            if SOURCELESS and fc != 'T':
                continue
            a1n = admin1_name(c, a1) if SOURCELESS else None
            if SOURCELESS and not a1n:
                continue
            pools.append((v, f'{FCL[fc]} in {a1n}, {country_name(c) if SOURCELESS else CT.get(c, c)}' if a1n else
                             f'{FCL[fc]} recorded in administrative division {a1} of '
                             f'{CT.get(c, c)}'))
    aggs = [('dem_gt', .45, 'dem', 'how many of them lie at a modelled terrain elevation '
                            'greater than {t} metres', 'how many of them lie at a modelled '
                            'terrain elevation of {t} metres or higher'),
            ('dem_lt', .30, 'dem', 'how many of them lie at a modelled terrain elevation '
                            'below {t} metres', 'how many of them lie at a modelled terrain '
                            'elevation of at most {t} metres'),
            ]
    blind = ('the large openly licensed global gazetteer of geographical names (the one that '
             'gives every entry a numeric identifier, a feature class and code, an '
             'administrative division, and a modelled terrain elevation)')
    if BLIND_SHORT: blind = 'the large openly licensed global gazetteer of geographical names'
    return pools, aggs, blind


SOURCELESS = _args.sourceless == 1

def _admin1_names():
    """Map GeoNames admin1 codes to real names (HR.17 -> Virovitica-Podravina).

        When the question names no database, the real place name is required: a
        code such as "administrative division 17 of HR" is specific to GeoNames and
        means nothing in any other source.
    """
    f = _fetch(f'{SCR}/admin1CodesASCII.txt',
               'https://download.geonames.org/export/dump/admin1CodesASCII.txt')
    d = {}
    for l in open(f, encoding='utf-8'):
        p = l.rstrip('\n').split('\t')
        if len(p) >= 2: d[p[0]] = p[1]
    return d

def _glotto_names():
    """Map a glottocode to its family name (mixe1284 -> Mixe-Zoque)."""
    import csv as _csv
    f = _fetch(f'{SCR}/glottolog_languages.csv',
               'https://raw.githubusercontent.com/glottolog/glottolog-cldf/master/cldf/languages.csv')
    with open(f, encoding='utf-8') as fh:
        return {r['ID']: r['Name'] for r in _csv.DictReader(fh)}

_A1 = None
_GL = None
def admin1_name(cc, code):
    global _A1
    if _A1 is None: _A1 = _admin1_names()
    return _A1.get(f'{cc.upper()}.{code}')

def glotto_name(gid):
    global _GL
    if _GL is None: _GL = _glotto_names()
    return _GL.get(gid)

IOC_DELTA = {
 'ALG':'Algeria','ANG':'Angola','ARU':'Aruba','BAH':'Bahamas','BAN':'Bangladesh',
 'BAR':'Barbados','BRU':'Brunei','BUL':'Bulgaria','BUR':'Burkina Faso','CAM':'Cambodia',
 'CAY':'Cayman Islands','CGO':'Congo','CHA':'Chad','CHI':'Chile','CRC':'Costa Rica',
 'CRO':'Croatia','DEN':'Denmark','ESA':'El Salvador','FIJ':'Fiji','GAM':'Gambia',
 'GBS':'Guinea-Bissau','GEQ':'Equatorial Guinea','GER':'Germany','GRE':'Greece',
 'GUA':'Guatemala','HAI':'Haiti','HON':'Honduras','INA':'Indonesia','IRI':'Iran',
 'IRQ':'Iraq','ISV':'United States Virgin Islands','IVB':'British Virgin Islands',
 'KSA':'Saudi Arabia','KUW':'Kuwait','LAT':'Latvia','LBA':'Libya','LES':'Lesotho',
 'LIB':'Lebanon','MAD':'Madagascar','MAS':'Malaysia','MAW':'Malawi','MGL':'Mongolia',
 'MON':'Monaco','MRI':'Mauritius','MTN':'Mauritania','MYA':'Myanmar','NCA':'Nicaragua',
 'NED':'Netherlands','NEP':'Nepal','NGR':'Nigeria','NIG':'Niger','OMA':'Oman',
 'PAR':'Paraguay','PHI':'Philippines','PLE':'Palestine','PNG':'Papua New Guinea',
 'POR':'Portugal','PUR':'Puerto Rico','RSA':'South Africa','SEY':'Seychelles',
 'SIN':'Singapore','SLO':'Slovenia','SRI':'Sri Lanka','SUD':'Sudan','SUI':'Switzerland',
 'SUR':'Suriname','TAN':'Tanzania','TOG':'Togo','TPE':'Chinese Taipei',
 'TRI':'Trinidad and Tobago','UAE':'United Arab Emirates','URU':'Uruguay','VIE':'Vietnam',
 'VIN':'Saint Vincent and the Grenadines','ZAM':'Zambia','ZIM':'Zimbabwe',
}

_CN = None
def _country_map():
    """Map ISO2 codes to country names, built from the GeoNames countryInfo.txt."""
    f = _fetch(f'{SCR}/countryInfo.txt',
               'https://download.geonames.org/export/dump/countryInfo.txt')
    d = {}
    for l in open(f, encoding='utf-8'):
        if l.startswith('#') or not l.strip(): continue
        p = l.rstrip('\n').split('\t')
        if len(p) > 4 and p[0] and p[4]:
            d[p[0]] = p[4]
            if p[1]: d[p[1]] = p[4]
    return d

def country_name(cc):
    """Resolve a country code to its name, returning the code unchanged if unknown."""
    global _CN
    k = (cc or '').upper()
    if k in IOC_DELTA: return IOC_DELTA[k]
    if _CN is None: _CN = _country_map()
    return _CN.get(k, cc)

_BBOX = None
def _country_bbox(name):
    """Bounding box of a country, derived from its cities15000 coordinates plus a margin.

        Sufficient for deciding whether a coordinate falls inside a country without
        downloading boundary data.
    """
    global _BBOX
    if _BBOX is None:
        _BBOX = {}
        acc = {}
        for c in cities():
            nm = country_name(c['cc'])
            a = acc.setdefault(nm, [90, -90, 180, -180])
            a[0] = min(a[0], c['lat']); a[1] = max(a[1], c['lat'])
            a[2] = min(a[2], c['lon']); a[3] = max(a[3], c['lon'])
        for k, a in acc.items():
            _BBOX[k] = (a[0] - 2, a[1] + 2, a[2] - 2, a[3] + 2)
    return _BBOX.get(name)

def _lead(blind, desc):
    """Opening clause of a question. Names no data source when --sourceless is set."""
    return f'Consider all {desc}.' if SOURCELESS else f'Using {blind}, consider all {desc}.'

def _plural(t):
    """Fix "more than 1 authors" once a question has been assembled."""
    return re.sub(r'\b1 ([a-z]+)s\b', lambda m: '1 ' + m.group(1), t)


_SRC_NAMES = {}

_EXCLUDE_KEYS = set()
for _xf in [x.strip() for x in os.environ.get('EXCLUDE', '').split(',') if x.strip()]:
    try:
        for _l in open(_xf, encoding='utf-8'):
            _m = (json.loads(_l).get('meta') or {})
            _EXCLUDE_KEYS.add((_m.get('pool_desc'), _m.get('agg'), str(_m.get('threshold'))))
    except Exception as _e:
        print(f'[rank] could not read EXCLUDE file {_xf}: {_e}', flush=True)
if _EXCLUDE_KEYS:
    print(f'[rank] loaded {len(_EXCLUDE_KEYS)} (pool, metric, threshold) keys to exclude', flush=True)
NAMEF_GLOBAL = {'AIR': 'name', 'GEO': 'name', 
                'LANG': 'name', 'OA': 'name', 'PWR': 'name'}


def _ans_searchable(a):
    """Whether an answer is phrased in a way a person could search for.

        Legitimate identifiers such as an RFC number or a three-letter airport code
        are accepted.
    """
    a = (a or '').strip()
    if not a:
        return False
    if a.isupper() and len(a) > 3 and not re.match(r'^(RFC\d+|[A-Z]{2,4}\d*)$', a):
        return False
    return True


FIELD_DESC = {'dem': 'modelled terrain elevation',
              'capacity_mw': 'installed capacity in megawatts',
              'elevation_m': 'summit elevation in metres',
              'commissioning_year': 'year of commissioning',
              'latitude': 'recorded reference latitude',
              'longitude': 'recorded reference longitude',
              'elevation_ft': 'published field elevation',
              'works_count': 'number of indexed works',
              'cited_by_count': 'total citation count'}

DISCRETE = {'dem': 1, 'commissioning_year': 1,
            
            
            'works_count': 1, 'cited_by_count': 1}

SOURCES = {'AIR': src_air, 'GEO': src_geo, 'LANG': src_lang,
           'OA': src_oa, 'PWR': src_pwr}
WANT = [x.strip().upper() for x in _args.src.split(',') if x.strip()]
_unknown = [w for w in WANT if w not in SOURCES]
if _unknown:
    raise SystemExit(f'unknown source {_unknown}; available: {sorted(SOURCES)}')


def build(pool, desc, aggs, blind, src):
    """Build one question: choose a pool, an answer shape, and a threshold.

        Returns None when the chosen combination does not yield a well-posed item,
        for example when the extremum has no clear margin over the runner-up, or
        when the answer is not phrased in a searchable form.
    """
    # Fields whose values do not agree across independent sources, so a solver
    # consulting a different source cannot reproduce the answer. 'dem' is the
    # SRTM-modelled elevation, which only GEO states explicitly as such.
    _BAD_FIELDS = {'cyear'} if src == 'GEO' else {'dem', 'cyear'}
    _COUNT_OK   = {'elev', 'pages', 'mag'}
    def _fld(a):
        return re.sub(r'_(gt|lt)$', '', str(a[0]))
    aggs = [a for a in aggs if _fld(a) not in _BAD_FIELDS]
    if not aggs:
        return None, 'all_fields_banned'

    SHAPE_W = {
        'AIR':  {'count_thresh': .15, 'extremum': .35, 'join_count': .00, 'ext_join': .50, 'conj': .00},
        'LANG': {'count_thresh': .10, 'extremum': .40, 'join_count': .00, 'ext_join': .50, 'conj': .00},
        'PWR':  {'count_thresh': .30, 'extremum': .70, 'join_count': .00, 'ext_join': .00, 'conj': .00},
        'GEO':  {'count_thresh': .30, 'extremum': .70, 'join_count': .00, 'ext_join': .00, 'conj': .00},
        'OA':   {'count_thresh': .30, 'extremum': .70, 'join_count': .00, 'ext_join': .00, 'conj': .00},
    }
    _w = SHAPE_W.get(src, {'count_thresh': .40, 'extremum': .35, 'join_count': .25})
    _w = {k: v for k, v in _w.items() if v > 0}
    if not _w:
        raise SystemExit(f'source {src} has every answer shape weighted 0 in SHAPE_W; '
                         f'give it a non-zero weight to use it')
    shape = rng.choices(list(_w), list(_w.values()))[0]
    NAMEF = {'AIR': 'name', 'GEO': 'name', 
             'LANG': 'name', 'OA': 'name', 'PWR': 'name'}

    if shape == 'extremum':
        cand = [a for a in aggs if a[2]]
        if not cand:
            return None, 'no_numeric'
        A = rng.choice(cand)
        nf = NAMEF.get(src)
        if not nf or not all((r.get(nf) or '').strip() for r in pool):
            return None, 'no_name'
        vals = [(fnum(r.get(A[2])), r) for r in pool if fnum(r.get(A[2])) is not None]
        if len(vals) < len(pool) * 0.9:
            return None, 'sparse_%s' % A[2]
        hi = rng.choice([True, False])
        vals.sort(key=lambda kv: -kv[0] if hi else kv[0])
        top, sec = vals[0], vals[1]
        span = (max(v for v, _ in vals) - min(v for v, _ in vals)) or 1
        _d = abs(top[0] - sec[0])
        _st = DISCRETE.get(A[2])
        if _st is not None:
            if _d < 2 * _st:
                return None, 'margin'
        elif _d < 0.03 * span:
            return None, 'margin'
        ans = (top[1].get(nf) or '').strip()
        if len(ans) < 3 or len(ans) > 90:
            return None, 'name_bad'
        word = 'largest' if hi else 'smallest'
        _fd = (A[5] if len(A) > 5 and A[5] else
               (A[4] if len(A) > 4 and A[4] and '{t}' not in str(A[4]) else
                FIELD_DESC.get(A[2])))
        if not _fd:
            return None, 'no_fielddesc_%s' % A[2]
        if A[2] == 'birthday':
            tail = f'which one was born {"most recently" if hi else "earliest"}'
        else:
            tail = f'which one has the {word} {_fd}'
        q = (_lead(blind, desc) + f' Among those, {tail}? '
             + ('Give only its name.' if SOURCELESS else
                'Give only its name, exactly as recorded in that dataset.'))
        if not _ans_searchable(ans):
            return None, 'ans_allcaps'
        if _SRC_NAMES.get(src, {}).get((ans or '').strip(), 0) > 1:
            return None, 'ans_ambiguous'
        meta = {'pool_desc': desc, 'n_pool': len(pool), 'agg': f'ext_{A[0]}_{"max" if hi else "min"}',
                'threshold': None, 'as_of': AS_OF, 'runner_up': (sec[1].get(nf) or '')[:60]}
        return dict(question=_plural(q), ground_truth=ans, answer_type='exactMatch', tolerance=0,
                    tree={'steps': [
                        {'description': f'pool: {desc} ({len(pool)} records)', 'critical': True,
                         'expected': len(pool)},
                        {'description': f'take the {word} by {A[2]}', 'critical': True, 'expected': ans},
                        {'description': f'Final answer (name): {ans}', 'critical': True, 'expected': ans}]},
                    meta=meta), 'ok'

    if shape == 'conj':
        num = [a for a in aggs if a[2]]
        if len(num) < 2:
            return None, 'conj_need2'
        A = B = None
        for _ in range(6):
            A, B = rng.sample(num, 2)
            if A[2] != B[2]: break
        if A is None or A[2] == B[2]:
            return None, 'conj_samefield'
        def pick(Ax):
            vs = sorted(v for v in (fnum(r.get(Ax[2])) for r in pool) if v is not None)
            if len(vs) < len(pool) * 0.9:
                return None, None, None
            raw = vs[int(len(vs) * rng.choice([0.35, 0.5, 0.65]))]
            st = DISCRETE.get(Ax[2])
            if st is None:
                t = tshow = nice(raw)
            else:
                g = round(raw / st) * st
                t = round(g + st / 2.0, 3)
                tshow = int(g) if st == 1 else round(g, 3)
            if not (vs[0] < t < vs[-1]):
                return None, None, None
            if st == 1 and tshow <= 0:
                return None, None, None
            gt = Ax[0].endswith('_gt')
            hit = {id(r) for r in pool if fnum(r.get(Ax[2])) is not None and
                   ((fnum(r[Ax[2]]) > t) if gt else (fnum(r[Ax[2]]) < t))}
            txt = Ax[3].format(t=tshow) if '{t}' in Ax[3] else Ax[3]
            txt = re.sub(r'^\s*how many of them\s+', '', txt)
            return t, hit, txt
        tA, hA, dA = pick(A)
        tB, hB, dB = pick(B)
        if tA is None or tB is None:
            return None, 'conj_sparse'
        if len(hA) < 3 or len(hB) < 3:
            return None, 'conj_thin'
        n = len(hA & hB)
        if n > 0.85 * len(pool):
            return None, 'conj_trivial'
        q = _plural(_lead(blind, desc) + f' Among those, how many satisfy both '
                    f'of the following: {dA}, and {dB}? Answer with a single integer; answer 0 '
                    f'if none of them satisfy both conditions.')
        return dict(question=_plural(q), ground_truth=str(n), answer_type='numeric', tolerance=0,
                    tree={'steps': [
                        {'description': f'pool: {desc} ({len(pool)})', 'critical': True,
                         'expected': len(pool)},
                        {'description': f'condition A ({A[2]} @ {tA}) -> {len(hA)} hits',
                         'critical': True, 'expected': len(hA)},
                        {'description': f'condition B ({B[2]} @ {tB}) -> {len(hB)} hits',
                         'critical': True, 'expected': len(hB)},
                        {'description': f'Final answer (conjunction): {n}', 'critical': True,
                         'expected': str(n)}]},
                    meta={'pool_desc': desc, 'n_pool': len(pool), 'agg': 'conj',
                          'threshold': [tA, tB], 'as_of': AS_OF,
                          'is_trap': n == 0, 'hits_a': len(hA), 'hits_b': len(hB)}), 'ok'

    if shape == 'ext_join':
        C = cities()
        cand = [a for a in aggs if a[2]]
        if not cand:
            return None, 'no_numeric'
        A = rng.choice(cand)
        nf = NAMEF.get(src)
        vals = [(fnum(r.get(A[2])), r) for r in pool if fnum(r.get(A[2])) is not None]
        if len(vals) < max(8, len(pool) * 0.7):
            return None, 'sparse_%s' % A[2]
        hi = rng.choice([True, False])
        vals.sort(key=lambda kv: -kv[0] if hi else kv[0])
        top, sec = vals[0], vals[1]
        span = (max(v for v, _ in vals) - min(v for v, _ in vals)) or 1
        _ABS = {'latitude': 0.3, 'longitude': 0.3}
        _d = abs(top[0] - sec[0])
        if A[2] in _ABS:
            if _d < _ABS[A[2]]:
                return None, 'margin'
        elif _d < 0.05 * span:
            return None, 'margin'
        _miss = (len(pool) - len(vals)) / max(1, len(pool))
        _partial = (' (considering only those for which that value is recorded)'
                    if _miss >= 0.05 else '')
        word = 'largest' if hi else 'smallest'
        fdesc = (A[5] if len(A) > 5 and A[5] else
                 (A[4] if len(A) > 4 and A[4] and '{t}' not in str(A[4]) else
                  FIELD_DESC.get(A[2])))
        if not fdesc:
            return None, 'no_fielddesc_%s' % A[2]
        tgt = top[1]

        if src == 'AIR':
            muni = (tgt.get('municipality') or '').strip()
            if not muni:
                return None, 'no_muni'
            cc = tgt.get('iso_country')
            hits = [c for c in C if c['cc'] == cc and muni in (c['name'], c['ascii'])]
            if len(hits) != 1:
                return None, 'city_%d' % len(hits)
            if SOURCELESS:
                ans = hits[0]['name']
                tail = ('then name the municipality that this one facility serves')
            else:
                ans = str(hits[0]['pop'])
                tail = ('then, for the municipality that this one facility serves, report the '
                        'population recorded for that municipality in the standard openly '
                        'licensed global gazetteer of populated places')
            hop3 = f'municipality {muni} -> population {ans}'
        elif src == 'PWR':
            own = (tgt.get('owner') or '').strip()
            if not own or len(own) < 3:
                return None, 'no_owner'
            if own.lower() in ('unknown', 'n/a', 'na', 'none', 'not available', '-'):
                return None, 'owner_placeholder'
            ans = own
            tail = 'then name the company recorded as its owner'
            hop3 = f'owner -> {ans}'
        elif src == 'LANG':
            la, lo = fnum(tgt.get('latitude')), fnum(tgt.get('longitude'))
            if la is None or lo is None:
                return None, 'no_coord'
            _cn = desc.rsplit(' in ', 1)[1].strip().rstrip('.') if ' in ' in desc else None
            if _cn:
                _cn = re.split(r'\s+(?:during|worldwide|in)\s+', _cn)[0].strip().rstrip('.')
            _bb = _country_bbox(_cn) if _cn else None
            if _bb and not (_bb[0] <= la <= _bb[1] and _bb[2] <= lo <= _bb[3]):
                return None, 'outside_country'
            near = sorted(((_haversine(la, lo, c['lat'], c['lon']), c) for c in C
                           if c['pop'] > 100000), key=lambda kv: kv[0])[:2]
            if len(near) < 2 or near[0][0] > 500:
                return None, 'no_city'
            if near[1][0] - near[0][0] < 0.08 * max(near[0][0], 1):
                return None, 'city_margin'
            ans = near[0][1]['name']
            tail = ('then name the nearest city with more than 100,000 inhabitants'
                    if SOURCELESS else
                    'then identify the nearest city with a population above 100,000 as '
                    'recorded in the standard openly licensed global gazetteer of populated '
                    'places, and give that city\'s name')
            hop3 = f'nearest big city -> {ans} ({near[0][0]:.0f} km)'
        else:
            shape = 'count_thresh'
            ans = None

        if ans is not None and not _ans_searchable(ans):
            return None, 'ans3_allcaps'
        if ans is not None:
            q = (_lead(blind, desc) + f' Among those, first take the one with '
                 f'the {word} {fdesc}{_partial}; {tail}.')
            return dict(question=_plural(q), ground_truth=ans,
                        answer_type='numeric' if ans.isdigit() else 'exactMatch',
                        tolerance=0,
                        tree={'steps': [
                            {'description': f'pool from source A: {desc} ({len(pool)})',
                             'critical': True, 'expected': len(pool)},
                            {'description': f'uniquify by {word} {fdesc}', 'critical': True,
                             'expected': (tgt.get(nf) or '?') if nf else '?'},
                            {'description': f'cross-source lookup: {hop3}', 'critical': True,
                             'expected': ans},
                            {'description': f'Final answer: {ans}', 'critical': True,
                             'expected': ans}]},
                        meta={'pool_desc': desc, 'n_pool': len(pool),
                              'agg': f'extjoin_{A[0]}_{"max" if hi else "min"}',
                              'threshold': None, 'as_of': AS_OF, 'cross_source': True,
                              'hops': 3, 'uniquified': (tgt.get(nf) or '') if nf else ''}), 'ok'

    if shape == 'join_count':
        C = cities()
        if src == 'AIR':
            muni = [(r.get('municipality') or '').strip() for r in pool]
            if sum(1 for m in muni if m) < len(pool) * 0.8:
                return None, 'no_muni'
            cc = pool[0].get('iso_country')
            idx = {}
            for c in C:
                if c['cc'] == cc:
                    idx.setdefault(c['name'], c['pop'])
                    idx.setdefault(c['ascii'], c['pop'])
            t = rng.choice([25000, 50000, 100000, 250000])
            hit = [r for r, m in zip(pool, muni) if idx.get(m, 0) > t]
            tail = (f'how many of them serve a town with more than {t:,} inhabitants'
                    if SOURCELESS else
                    f'how many of them serve a municipality that is listed, in the standard '
                    f'openly licensed global gazetteer of populated places, with a population '
                    f'above {t:,}')
        elif src == 'LANG':
            rad = rng.choice([50, 100, 200])
            t = rng.choice([50000, 100000, 500000])
            big = [c for c in C if c['pop'] > t]
            hit = []
            for r in pool:
                la, lo = fnum(r.get('latitude')), fnum(r.get('longitude'))
                if la is None or lo is None:
                    return None, 'no_coord'
                if any(_haversine(la, lo, c['lat'], c['lon']) <= rad for c in big):
                    hit.append(r)
            tail = (f'how many of them have a reference location within {rad} km of a city '
                    f'with more than {t:,} inhabitants' if SOURCELESS else
                    f'how many of them have a reference location within {rad} km of a city '
                    f'that is listed, in the standard openly licensed global gazetteer of '
                    f'populated places, with a population above {t:,}')
        else:
            shape = 'count_thresh'
            hit = None
        if hit is not None:
          n = len(hit)
          if n in (0, 1) or n == len(pool) or not (0.1 <= n / len(pool) <= 0.9):
            return None, 'degenerate_%d' % n
          q = (_lead(blind, desc) + f' Among those, {tail}? '
             f'Answer with a single integer.')
          return dict(question=_plural(q), ground_truth=str(n), answer_type='numeric', tolerance=0,
                    tree={'steps': [
                        {'description': f'pool from source A: {desc} ({len(pool)})',
                         'critical': True, 'expected': len(pool)},
                        {'description': f'for each item, look up the joined attribute ({src} -> cities)',
                         'critical': True, 'expected': 'join'},
                        {'description': f'Final answer (count): {n}', 'critical': True,
                         'expected': str(n)}]},
                    meta={'pool_desc': desc, 'n_pool': len(pool), 'agg': f'join_{src}',
                          'threshold': t, 'as_of': AS_OF, 'cross_source': True}), 'ok'

    _cnt_cand = [a for a in aggs if a[2] and _fld(a) in _COUNT_OK]
    if not _cnt_cand:
        return None, 'no_countable_field'
    A = rng.choice(_cnt_cand)
    if not A[2]:
        if A[0] == 'sched':
            hit = [r for r in pool if (r.get('scheduled_service') or '').strip().lower() == 'yes']
        else:
            hit = [r for r in pool if (r.get('iata_code') or '').strip()]
        t, tail = None, A[3]
    else:
        vals = sorted(v for v in (fnum(r.get(A[2])) for r in pool) if v is not None)
        if len(vals) < len(pool) * 0.9:
            return None, 'sparse_%s' % A[2]
        gt = A[0].endswith('_gt')
        t = hit = None
        for _q in rng.sample([0.3, 0.4, 0.45, 0.55, 0.6, 0.7], 6):
            _t = nice(vals[int(len(vals) * _q)])
            _eq = [r for r in pool if fnum(r.get(A[2])) == _t]
            if len(_eq) > max(1, 0.05 * len(pool)):
                continue
            _hit = [r for r in pool if fnum(r.get(A[2])) is not None and
                    ((fnum(r[A[2]]) > _t) if gt else (fnum(r[A[2]]) < _t))]
            _n = len(_hit)
            if _n in (0, 1) or _n == len(pool) or not (0.1 <= _n / len(pool) <= 0.9):
                continue
            t, hit = _t, _hit
            break
        if t is None:
            return None, 'boundary'
        tail = A[3].format(t=t)
    n = len(hit)
    if n in (0, 1) or n == len(pool) or not (0.1 <= n / len(pool) <= 0.9):
        return None, 'degenerate_%d' % n
    q = (_lead(blind, desc) + f' Among those, {tail}? '
         f'Answer with a single integer.')
    return dict(question=_plural(q), ground_truth=str(n), answer_type='numeric', tolerance=0,
                tree={'steps': [
                    {'description': f'pool: {desc} ({len(pool)})', 'critical': True,
                     'expected': len(pool)},
                    {'description': f'count with threshold {t}', 'critical': True, 'expected': n},
                    {'description': f'Final answer (count): {n}', 'critical': True,
                     'expected': str(n)}]},
                meta={'pool_desc': desc, 'n_pool': len(pool), 'agg': A[0], 'threshold': t,
                      'as_of': AS_OF}), 'ok'


def main():
    """Allocate a quota per source and generate questions until --n is reached."""
    ALL = []
    for name in WANT:
        if name not in SOURCES:
            print(f'[rank] unknown source {name}, skipped', flush=True)
            continue
        pools, aggs, blind = SOURCES[name]()
        use = [(p, d) for p, d in pools if POOL_LO <= len(p) <= POOL_HI]
        print(f'[rank] {name}: {len(pools)} pools, {len(use)} within size {POOL_LO}-{POOL_HI}', flush=True)
        for p, d in use:
            ALL.append((name, p, d, aggs, blind))
        _nf = NAMEF_GLOBAL.get(name)
        if _nf:
            _seen = set()
            _cnt = collections.Counter()
            for _p, _d in pools:
                for _r in _p:
                    if id(_r) in _seen:
                        continue
                    _seen.add(id(_r))
                    _v = (_r.get(_nf) or '').strip()
                    if _v:
                        _cnt[_v] += 1
            _SRC_NAMES[name] = _cnt
    REPEAT = int(os.environ.get('REPEAT', '3'))
    ALL = ALL * REPEAT
    rng.shuffle(ALL)
    seen_pool = set()
    stat = collections.Counter()
    out = []
    if os.path.exists(OUT):
        for l in open(OUT):
            d = json.loads(l)
            out.append(d)
            seen_pool.add((d['meta']['pool_desc'], d['meta']['agg']))

    SHARE = {'AIR': .20, 'LANG': .18, 'GEO': .10, 'OA': .12, 'PWR': .12}
    _sh = {k: v for k, v in SHARE.items() if k in WANT}
    _tot = sum(_sh.values()) or 1
    per = {n: 0 for n in WANT}
    for _d in out:
        _w = _d.get('walk')
        if _w in per:
            per[_w] += 1
    CAP = {n: max(1, int(N * _sh.get(n, 1.0 / max(1, len(WANT))) / _tot)) for n in WANT}
    print(f'[rank] existing per source {({k: v for k, v in per.items() if v})}', flush=True)
    print(f'[rank] quota per source {CAP} (total {sum(CAP.values())})', flush=True)
    print(f'[rank] {len(ALL)} (pool, round) candidates', flush=True)

    print(f'[rank] {len(out)} questions already present', flush=True)

    for name, pool, desc, aggs, blind in ALL:
        if len(out) >= N:
            break
        if per[name] >= CAP[name]:
            continue
        pass
        res, why = build(pool, desc, aggs, blind, name)
        stat[f'{name}:{why}'] += 1
        if not res:
            continue
        if _EXCLUDE_KEYS:
            _m = res.get('meta') or {}
            if (_m.get('pool_desc'), _m.get('agg'), str(_m.get('threshold'))) in _EXCLUDE_KEYS:
                stat[f'{name}:xbatch_dup'] += 1
                continue
        _norm = re.sub(r'^.*? in ', '', desc)
        _thr = str(res['meta'].get('threshold'))
        key = (desc, res['meta']['agg'])
        key2 = (name, _norm, res['meta']['agg'], _thr)
        if key in seen_pool or key2 in seen_pool:
            stat[f'{name}:dup'] += 1
            continue
        res['id'] = f'{name.lower()}-' + hashlib.md5((desc + res['meta']['agg'] + VER).encode()).hexdigest()[:8]
        res['walk'] = name
        res['blind'] = True
        res['ver'] = VER
        res['meta']['src'] = name
        out.append(res)
        seen_pool.add((desc, res['meta']['agg']))
        seen_pool.add(key2)
        per[name] += 1
        with open(OUT, 'a') as f:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
        print(f'{len(out):>3} {res["id"]} [{name}] ans={res["ground_truth"]:>4} | {desc[:58]}', flush=True)

    print(f'DONE  {len(out)} questions, per source {per}', flush=True)
    print(f'  rejections {dict(stat)}', flush=True)



if __name__ == '__main__':
    main()
import re, csv, glob, os
from collections import defaultdict, Counter
import openpyxl

CHAR_MAP = {
    '+d': 'ð', '+D': 'Ð',
    '+a': 'æ', '+A': 'Æ',
    '+t': 'þ', '+T': 'Þ',
    '+e': 'æ',   # rare alternate escape observed in the corpus, decodes the same way
}
ESCAPE_RE = re.compile('|'.join(re.escape(k) for k in CHAR_MAP))

def decode(word):
    if not word:
        return word
    return ESCAPE_RE.sub(lambda m: CHAR_MAP[m.group(0)], word)

# ---- Load lemma list ----
LEMMA_PATH = '/mnt/user-data/uploads/varioe_lemmas.xlsx'
wb = openpyxl.load_workbook(LEMMA_PATH, read_only=True, data_only=True)
ws = wb['Sheet1']
rows = list(ws.iter_rows(values_only=True))
header, data = rows[0], rows[1:]
# header: orig_word, lemma, pos, pos_cat, tags, type

# exact-case lookup, restricted to noun entries (pos == 'N'), keyed by stripped orig_word
noun_lookup = defaultdict(list)      # word -> [(lemma, tags_str)]
noun_lookup_ci = defaultdict(list)   # lowercased word -> [(lemma, tags_str)]  (case-insensitive fallback)

for r in data:
    orig, lemma, pos, pos_cat, tags, typ = r
    if orig is None:
        continue
    key = orig.lstrip('$')
    if pos == 'N':
        noun_lookup[key].append((lemma, tags or ''))
        noun_lookup_ci[key.lower()].append((lemma, tags or ''))

def lookup_lemma(word, tag):
    """word: decoded surface form; tag: e.g. 'N^A'. Returns (lemma, match_type)."""
    word = word.lstrip('$')  # '$' marks an editorial/emendation mark in some YCOE tokens
    cands = noun_lookup.get(word)
    match_type = 'exact'
    if not cands:
        cands = noun_lookup_ci.get(word.lower())
        match_type = 'case-insensitive'
    if not cands:
        return '', 'unmatched'
    if len(cands) == 1:
        return cands[0][0], match_type
    # disambiguate by exact tag match against the row's attested tag set
    for lemma, tags in cands:
        if tag and tag in tags:
            return lemma, match_type + '+tag-disambiguated'
    # fall back to first candidate, flag as ambiguous
    return cands[0][0], match_type + '+ambiguous(' + str(len(cands)) + ')'

# ---- Process each per-text noun CSV ----
files = sorted(glob.glob('/home/claude/extraction/*_nouns.csv'))
files = [f for f in files if not os.path.basename(f).startswith('all_texts')]

match_stats = Counter()
all_rows = []

for path in files:
    with open(path, encoding='utf-8') as f:
        rows_in = list(csv.DictReader(f))
    out_rows = []
    for r in rows_in:
        decoded_noun = decode(r['noun'])
        lemma, match_type = lookup_lemma(decoded_noun, r['noun_tag'])
        match_stats[match_type.split('+')[0]] += 1
        r['noun_decoded'] = decoded_noun
        r['lemma'] = lemma
        r['lemma_match'] = match_type
        r['det_decoded'] = decode(r['det'])
        r['quantifier_decoded'] = decode(r['quantifier'])
        r['numeral_decoded'] = decode(r['numeral'])
        out_rows.append(r)
        all_rows.append(r)
    outname = path.replace('_nouns.csv', '_nouns_lemmatized.csv')
    fieldnames = list(out_rows[0].keys())
    with open(outname, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

with open('/home/claude/extraction/all_texts_nouns_lemmatized.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    writer.writeheader()
    writer.writerows(all_rows)

print('Lemma match stats (noun tokens):')
total = sum(match_stats.values())
for k, v in match_stats.most_common():
    print(f'  {k}: {v} ({v/total*100:.1f}%)')
print('TOTAL:', total)

# distinct lemmas after matching
lemmas = set(r['lemma'] for r in all_rows if r['lemma'])
print('Distinct lemmas recovered:', len(lemmas))

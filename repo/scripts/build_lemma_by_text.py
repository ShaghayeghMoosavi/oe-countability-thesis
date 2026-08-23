import pandas as pd

MIN_FREQ_OVERALL = 10   # same threshold used for the combined lemma set
MIN_FREQ_CELL = 3       # below this, a lemma-in-text proportion is flagged as low-confidence

df = pd.read_csv('/home/claude/extraction/final_all_texts.csv')
df = df[df['lemma'].notna() & (df['lemma'] != '')].copy()

df['has_det'] = df['det_decoded'].notna() & (df['det_decoded'].astype(str).str.len() > 0)
df['has_quant'] = df['quantifier_decoded'].notna() & (df['quantifier_decoded'].astype(str).str.len() > 0)
df['has_num'] = df['numeral_decoded'].notna() & (df['numeral_decoded'].astype(str).str.len() > 0)
df['is_bare'] = df['bare'].astype(str).str.lower() == 'true'
df['is_genitive'] = df['noun_case'] == 'G'

# keep only lemmas that meet the overall (combined-corpus) frequency threshold,
# so this is the same 589-lemma set as the pooled feature vectors, just broken out by text
overall_counts = df.groupby('lemma').size()
kept_lemmas = overall_counts[overall_counts >= MIN_FREQ_OVERALL].index
df = df[df['lemma'].isin(kept_lemmas)].copy()

rows = []
for (lemma, text), g in df.groupby(['lemma', 'text']):
    n = len(g)
    rows.append({
        'lemma': lemma,
        'text': text,
        'n': n,
        'low_confidence': n < MIN_FREQ_CELL,
        'bare_proportion': round(g['is_bare'].sum() / n, 4),
        'determiner_proportion': round(g['has_det'].sum() / n, 4),
        'quantifier_proportion': round(g['has_quant'].sum() / n, 4),
        'numeral_proportion': round(g['has_num'].sum() / n, 4),
        'genitive_proportion': round(g['is_genitive'].sum() / n, 4),
    })

long_df = pd.DataFrame(rows).sort_values(['lemma', 'text']).reset_index(drop=True)
long_df.to_csv('/home/claude/extraction/lemma_by_text_features_long.csv', index=False)

print(f"Lemma x text rows: {len(long_df)}  ({long_df['lemma'].nunique()} lemmas x up to 10 texts)")
print(f"Low-confidence cells (n<{MIN_FREQ_CELL}): {long_df['low_confidence'].sum()} of {len(long_df)}")
print()

# wide pivot tables, one per feature, for direct across-text comparison
pivots = {}
for feat in ['n', 'bare_proportion', 'determiner_proportion', 'quantifier_proportion',
             'numeral_proportion', 'genitive_proportion']:
    pivots[feat] = long_df.pivot(index='lemma', columns='text', values=feat)

# order lemma rows by overall frequency (most frequent first), matching the pooled feature file
order = overall_counts.reindex(pivots['n'].index).sort_values(ascending=False).index
for feat in pivots:
    pivots[feat] = pivots[feat].reindex(order)

print("Example: 'dæg' (day) across texts —")
print(pivots['bare_proportion'].loc['dæg'].to_string())
print()
print("Example: 'cyning' (king) across texts —")
print(pivots['genitive_proportion'].loc['cyning'].to_string())

import pickle
with open('/home/claude/extraction/pivots.pkl', 'wb') as f:
    pickle.dump(pivots, f)

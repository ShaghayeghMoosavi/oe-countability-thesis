import pandas as pd
import numpy as np

MIN_FREQ = 10  # minimum total occurrences across the combined corpus to retain a lemma

df = pd.read_csv('/home/claude/extraction/final_all_texts.csv')
df = df[df['lemma'].notna() & (df['lemma'] != '')].copy()

# per-token booleans
df['has_det'] = df['det_decoded'].notna() & (df['det_decoded'].astype(str).str.len() > 0)
df['has_quant'] = df['quantifier_decoded'].notna() & (df['quantifier_decoded'].astype(str).str.len() > 0)
df['has_num'] = df['numeral_decoded'].notna() & (df['numeral_decoded'].astype(str).str.len() > 0)
df['is_bare'] = df['bare'].astype(str).str.lower() == 'true'
df['is_genitive'] = df['noun_case'] == 'G'

grouped = df.groupby('lemma')

feature_rows = []
for lemma, g in grouped:
    n = len(g)
    if n < MIN_FREQ:
        continue
    feature_rows.append({
        'lemma': lemma,
        'n_total': n,
        'n_texts': g['text'].nunique(),
        'texts': ';'.join(sorted(g['text'].unique())),
        'bare_proportion': round(g['is_bare'].sum() / n, 4),
        'determiner_proportion': round(g['has_det'].sum() / n, 4),
        'quantifier_proportion': round(g['has_quant'].sum() / n, 4),
        'numeral_proportion': round(g['has_num'].sum() / n, 4),
        'genitive_proportion': round(g['is_genitive'].sum() / n, 4),
    })

features = pd.DataFrame(feature_rows).sort_values('n_total', ascending=False).reset_index(drop=True)

print(f"Lemmas with >= {MIN_FREQ} occurrences in the combined corpus: {len(features)}")
print(f"(out of {df['lemma'].nunique()} distinct lemmas, {len(df)} total noun tokens)")
print()
print(features.head(15).to_string(index=False))

features.to_csv('/home/claude/extraction/lemma_feature_vectors.csv', index=False)

# quick distributional sanity checks
print()
print("Feature ranges:")
for col in ['bare_proportion','determiner_proportion','quantifier_proportion','numeral_proportion','genitive_proportion']:
    print(f"  {col}: min={features[col].min():.3f} max={features[col].max():.3f} mean={features[col].mean():.3f}")

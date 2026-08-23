import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

df = pd.read_csv('/home/claude/extraction/lemma_feature_vectors_clustered.csv').set_index('lemma')

FEATURES = ['bare_proportion', 'determiner_proportion', 'quantifier_proportion',
            'numeral_proportion', 'genitive_proportion']

# Standard dictionary glosses (Bosworth-Toller / Clark Hall) for the cherry-picked lemmas only.
GLOSS = {
    'hunig': 'honey', 'wisa': 'way/manner', 'ecnes': 'eternity', 'dohtor': 'daughter',
    'godspellere': 'evangelist', 'casere': 'emperor', 'sacerd': 'priest', 'heahengel': 'archangel',
    'abbod': 'abbot', 'gæst': 'spirit/soul', 'sunne': 'sun', 'soþfæstness': 'truth',
    'fæmne': 'virgin/woman', 'halignes': 'holiness', 'þrymm': 'glory',
    'scilling': 'shilling', 'halig': 'holy one/saint', 'winter': 'winter/year', 'middaneard': 'world',
    'monaþ': 'month', 'niht': 'night', 'gear': 'year',
    'gesceaft': 'creation', 'þing': 'thing',
    'man': 'man/one', 'cyning': 'king', 'god': 'god', 'dæg': 'day',
}

def label(lemma):
    g = GLOSS.get(lemma)
    return f"{lemma} '{g}'" if g else lemma

# ---------------- Selection of lemmas to cherry-pick ----------------
groups = {
    'extreme bare (mass-like)': ['hunig', 'wisa', 'ecnes', 'dohtor'],
    'extreme determiner (count-like)': ['godspellere', 'casere', 'sacerd', 'heahengel'],
    'high det + high genitive (DBSCAN cluster 1)': ['abbod', 'gæst', 'sunne', 'soþfæstness', 'fæmne'],
    'high genitive, other clusters (measure/formulaic)': ['scilling', 'halig', 'winter', 'middaneard'],
    'high numeral (time nouns)': ['monaþ', 'niht', 'gear'],
    'high quantifier': ['gesceaft', 'þing'],
    'high-frequency anchors': ['man', 'cyning', 'god', 'dæg'],
}
picked = [l for grp in groups.values() for l in grp]
picked_df = df.loc[picked]

# =====================================================================
# Figure 1: full-corpus scatter with cherry-picked lemmas labeled
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(df['bare_proportion'], df['determiner_proportion'],
           s=14, color='lightgray', alpha=0.5, zorder=1, label='all 589 lemmas')

group_colors = {
    'extreme bare (mass-like)': '#4C72B0',
    'extreme determiner (count-like)': '#DD8452',
    'high det + high genitive (DBSCAN cluster 1)': '#C44E52',
    'high genitive, other clusters (measure/formulaic)': '#8172B2',
    'high numeral (time nouns)': '#55A868',
    'high quantifier': '#937860',
    'high-frequency anchors': '#333333',
}

for grp, lemmas in groups.items():
    sub = df.loc[lemmas]
    sizes = 25 + sub['n_total'] * 0.6
    ax.scatter(sub['bare_proportion'], sub['determiner_proportion'],
               s=sizes, color=group_colors[grp], edgecolor='black', linewidth=0.6,
               zorder=3, label=grp)
    for lemma, row in sub.iterrows():
        ax.annotate(label(lemma), (row['bare_proportion'], row['determiner_proportion']),
                    xytext=(5, 4), textcoords='offset points', fontsize=8,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

ax.set_xlabel('Bare proportion (mass-like) →')
ax.set_ylabel('Determiner proportion (count-like) →')
ax.set_title('Selected Old English nouns by bare vs. determiner proportion\n(marker size = total occurrences; gray = full 589-lemma corpus)')
ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
plt.tight_layout()
plt.savefig('/home/claude/extraction/fig_cherry_picked_scatter.png', dpi=160)
plt.close()
print('saved fig_cherry_picked_scatter.png')

# =====================================================================
# Figure 2: heatmap of cherry-picked lemmas across all five features
# =====================================================================
heat_df = picked_df[FEATURES].copy()
heat_df.index = [label(l) for l in heat_df.index]
heat_df = heat_df.sort_values('genitive_proportion', ascending=False)

fig, ax = plt.subplots(figsize=(9, 10))
sns.heatmap(heat_df, annot=True, fmt='.2f', cmap='YlOrRd', vmin=0, vmax=1,
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'proportion'}, ax=ax)
ax.set_title('Countability profiles of selected Old English nouns\n(sorted by genitive proportion)')
ax.set_xlabel('')
ax.set_ylabel('')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('/home/claude/extraction/fig_cherry_picked_heatmap.png', dpi=160)
plt.close()
print('saved fig_cherry_picked_heatmap.png')

# =====================================================================
# Figure 3: grouped bar chart, DBSCAN cluster 1 vs. contrasting lemmas
# =====================================================================
contrast_lemmas = ['abbod', 'gæst', 'sunne', 'soþfæstness', 'fæmne', 'halignes', 'þrymm',  # cluster 1 (all 7)
                   'hunig', 'wisa',    # extreme cluster 0 (mass-like) contrast
                   'scilling', 'winter']  # high-genitive noise-category contrast

bar_df = df.loc[contrast_lemmas, FEATURES]
bar_df.index = [label(l) for l in bar_df.index]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(bar_df))
width = 0.15
feat_colors = ['#4C72B0', '#DD8452', '#55A868', '#8172B2', '#C44E52']
for i, feat in enumerate(FEATURES):
    ax.bar(x + (i - 2) * width, bar_df[feat], width, label=feat.replace('_proportion', ''), color=feat_colors[i])
ax.set_xticks(x)
ax.set_xticklabels(bar_df.index, rotation=35, ha='right')
ax.set_ylabel('Proportion')
ax.set_title('DBSCAN cluster 1 (n=7) vs. contrasting lemmas from cluster 0 and noise')
ax.axvline(6.5, color='black', linestyle=':', linewidth=1)
ax.text(3, 1.02, 'DBSCAN cluster 1', ha='center', fontsize=9, style='italic')
ax.text(9, 1.02, 'contrast lemmas', ha='center', fontsize=9, style='italic')
ax.legend(fontsize=8, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.25))
plt.tight_layout()
plt.savefig('/home/claude/extraction/fig_cluster1_contrast_bars.png', dpi=160)
plt.close()
print('saved fig_cluster1_contrast_bars.png')

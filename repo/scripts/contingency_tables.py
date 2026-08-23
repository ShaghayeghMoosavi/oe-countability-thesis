"""
contingency_tables.py

Token-level contingency-table / chi-square tests for co-occurrence
relationships between countability-relevant features, used in place of
Pearson correlation on lemma-aggregated proportions (see Section 3.4 /
3.3 "Why proportions" of the thesis for the rationale).

Input:  all_texts_nouns_lemmatized.csv (output of extract.py + lemmatize.py)
Output: printed contingency tables, chi-square statistics, and Cramer's V
        for each tested relationship.
"""
import pandas as pd
from scipy.stats import chi2_contingency

INPUT_CSV = "all_texts_nouns_lemmatized.csv"


def run_test(rowvar, colvar, rowname, colname, labels_row, labels_col):
    table = pd.crosstab(rowvar, colvar)
    table.index = labels_row
    table.columns = labels_col
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.values.sum()
    cramers_v = (chi2 / n) ** 0.5
    print(f"--- {rowname} x {colname} ---")
    print(table)
    print(f"Chi2={chi2:.2f}, df={dof}, p={p:.2e}, Cramer's V={cramers_v:.3f}")
    print()
    return table, chi2, p, cramers_v


def main():
    df = pd.read_csv(INPUT_CSV)

    is_gen = df["noun_case"] == "G"
    has_num = df["numeral_decoded"].notna()
    has_det = df["det_decoded"].notna()
    has_quant = df["quantifier_decoded"].notna()
    has_other = has_det | has_quant

    # Numeral vs. any other modifier (bare/numeral relationship, RQ1)
    run_test(has_num, has_other, "numeral", "other modifier",
              ["no numeral", "has numeral"], ["no other modifier", "has other modifier"])

    # Genitive vs. numeral (RQ2)
    run_test(is_gen, has_num, "genitive", "numeral",
              ["not genitive", "genitive"], ["no numeral", "has numeral"])

    # Genitive vs. determiner (RQ2)
    run_test(is_gen, has_det, "genitive", "determiner",
              ["not genitive", "genitive"], ["no determiner", "has determiner"])


if __name__ == "__main__":
    main()

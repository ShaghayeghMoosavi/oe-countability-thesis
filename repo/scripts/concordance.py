import sys, re, csv

def load_rows(lemma, text=None):
    with open('/home/claude/extraction/final_all_texts.csv', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['lemma'] == lemma and (text is None or r['text'] == text)]
    return rows

def get_sentence(text_file, sent_id):
    """Pull the raw .pos line(s) for a given sentence ID and render it readably
    (word/TAG pairs, special chars decoded), skipping the leading CODE line."""
    path = f'/mnt/user-data/uploads/{text_file}.pos'
    with open(path, encoding='utf-8') as f:
        content = f.read()
    # sentences are separated by blank lines; each ends with ..._ID
    blocks = content.split('\n\n')
    for b in blocks:
        if sent_id in b:
            return b.strip()
    return None

CHAR_MAP = {'+d':'ð','+D':'Ð','+a':'æ','+A':'Æ','+t':'þ','+T':'Þ','+e':'æ'}
ESCAPE_RE = re.compile('|'.join(re.escape(k) for k in CHAR_MAP))
def decode(w):
    return ESCAPE_RE.sub(lambda m: CHAR_MAP[m.group(0)], w) if w else w

def render(block):
    """Strip tags for a readable running line, and show tagged form beneath it."""
    tokens = re.findall(r'(\S+?)_([A-Za-z0-9^+$.,]+)', block)
    words = [decode(t[0].lstrip('$')) for t in tokens if not t[0].startswith('<')]
    tagged = [f"{decode(t[0].lstrip('$'))}_{t[1]}" for t in tokens if not t[0].startswith('<')]
    return ' '.join(words), ' '.join(tagged)

def main(lemma, text=None):
    rows = load_rows(lemma, text)
    if not rows:
        print(f"No rows found for lemma={lemma!r} text={text!r}")
        return
    print(f"Found {len(rows)} occurrence(s) of '{lemma}'" + (f" in {text}" if text else " across all texts"))
    print()
    for r in rows:
        print(f"--- {r['text']}  [{r['sent_id']}]  case={r['noun_case']}  det={r['det_decoded']}  quant={r['quantifier_decoded']}  num={r['numeral_decoded']} ---")
        block = get_sentence(r['text'], r['sent_id'])
        if block:
            plain, tagged = render(block)
            print("  plain: ", plain)
            print("  tagged:", tagged)
        else:
            print("  (sentence not found)")
        print()

if __name__ == '__main__':
    lemma = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else None
    main(lemma, text)

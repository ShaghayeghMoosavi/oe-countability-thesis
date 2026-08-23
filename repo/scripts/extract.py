import re, sys, csv, glob, os
from collections import namedtuple

Node = namedtuple("Node", ["label", "children"])  # children: list of Node or str (leaf word)

TOKEN_RE = re.compile(r'\(|\)|[^\s()]+')

def tokenize(text):
    return TOKEN_RE.findall(text)

def parse_one(tokens, i):
    """tokens[i] == '('  -> returns (Node, next_i)"""
    assert tokens[i] == '('
    i += 1
    if tokens[i] == '(':
        label = None  # anonymous group (used at the top sentence level)
    else:
        label = tokens[i]
        i += 1
    children = []
    while tokens[i] != ')':
        if tokens[i] == '(':
            child, i = parse_one(tokens, i)
            children.append(child)
        else:
            children.append(tokens[i])
            i += 1
    i += 1  # consume ')'
    return Node(label, children), i

def parse_forest(text):
    """A .psd file is a sequence of top-level parenthesized groups."""
    tokens = tokenize(text)
    i = 0
    n = len(tokens)
    forest = []
    while i < n:
        if tokens[i] == '(':
            node, i = parse_one(tokens, i)
            forest.append(node)
        else:
            i += 1  # skip stray tokens between groups, if any
    return forest

def case_of(tag):
    """Extract case suffix after ^ from a POS tag, e.g. N^A -> A. Returns '' if none."""
    if tag is None:
        return ''
    m = re.search(r'\^([A-Z])', tag)
    return m.group(1) if m else ''

def base_tag(tag):
    """Strip ^case suffix, e.g. N^A -> N."""
    if tag is None:
        return ''
    return tag.split('^')[0]

def find_id(node):
    """Find the (ID ...) leaf text within a top-level sentence group."""
    if node.label == 'ID' and node.children:
        return node.children[0]
    for c in node.children:
        if isinstance(c, Node):
            found = find_id(c)
            if found:
                return found
    return None

def is_np_label(label):
    return label is not None and (label == 'NP' or label.startswith('NP-'))

def extract_from_np(np_node, sent_id, filename, rows):
    """Given an NP node, find its head noun(s) among direct children,
    and any direct-child determiner / quantifier / numeral siblings."""
    det = []      # (word, case)
    quant = []    # (word, case)
    num = []      # (word, case)
    heads = []    # (word, tag)

    for child in np_node.children:
        if isinstance(child, Node):
            bt = base_tag(child.label)
            if bt == 'N':  # common noun head (excludes NR, NUM, NP, NX)
                word = child.children[0] if child.children else ''
                heads.append((word, child.label))
            elif bt == 'D':
                word = child.children[0] if child.children else ''
                det.append((word, case_of(child.label)))
            elif bt in ('Q', 'QR', 'QS'):
                word = child.children[0] if child.children else ''
                quant.append((word, case_of(child.label)))
            elif bt == 'NUM':
                word = child.children[0] if child.children else ''
                num.append((word, case_of(child.label)))
            elif bt == 'NUMP':
                # numeral phrase wrapper - look one level inside for the NUM
                for gc in child.children:
                    if isinstance(gc, Node) and base_tag(gc.label) == 'NUM':
                        word = gc.children[0] if gc.children else ''
                        num.append((word, case_of(gc.label)))
        # leaves that are plain strings (rare at this level) are ignored

    for word, tag in heads:
        rows.append({
            'text': filename,
            'sent_id': sent_id,
            'np_label': np_node.label,
            'noun': word,
            'noun_tag': tag,
            'noun_case': case_of(tag),
            'det': det[0][0] if det else '',
            'det_case': det[0][1] if det else '',
            'quantifier': quant[0][0] if quant else '',
            'quant_case': quant[0][1] if quant else '',
            'numeral': num[0][0] if num else '',
            'numeral_case': num[0][1] if num else '',
            'n_determiners': len(det),
            'n_quantifiers': len(quant),
            'n_numerals': len(num),
            'bare': (len(det) == 0 and len(quant) == 0 and len(num) == 0),
        })

def walk(node, sent_id, filename, rows):
    if isinstance(node, Node):
        if is_np_label(node.label):
            extract_from_np(node, sent_id, filename, rows)
        for c in node.children:
            walk(c, sent_id, filename, rows)

def process_file(path):
    filename = os.path.basename(path).replace('.psd', '')
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    forest = parse_forest(text)
    rows = []
    for top in forest:
        sent_id = find_id(top) or ''
        walk(top, sent_id, filename, rows)
    return rows

if __name__ == '__main__':
    files = sorted(glob.glob('/home/claude/extraction/*.psd'))
    all_rows = []
    summary = []
    for path in files:
        rows = process_file(path)
        all_rows.extend(rows)
        summary.append((os.path.basename(path), len(rows)))
        # per-text CSV
        outname = os.path.basename(path).replace('.psd', '_nouns.csv')
        with open(f'/home/claude/extraction/{outname}', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
                'text','sent_id','np_label','noun','noun_tag','noun_case','det','det_case',
                'quantifier','quant_case','numeral','numeral_case','n_determiners','n_quantifiers','n_numerals','bare'])
            writer.writeheader()
            writer.writerows(rows)

    with open('/home/claude/extraction/all_texts_nouns.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("Per-file noun-phrase counts:")
    for name, count in summary:
        print(f"  {name}: {count}")
    print(f"TOTAL: {len(all_rows)}")

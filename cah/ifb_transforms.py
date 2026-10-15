from pathlib import Path
import csv
import io
import os
import re
import string
import unicodedata
from collections import Counter
os.environ.setdefault('NLTK_DATA', str(Path(__file__).resolve().parents[1] / 'assets/nltk'))
import nltk
from nltk.tokenize import RegexpTokenizer
_WORD_TOK = RegexpTokenizer('\\w+')
_PUNCT = string.punctuation
_PUNCT_SPACE = _PUNCT + ' '
LAST = {}

def _official(iid, kw, text, prompt):
    from ifb_official import official
    return official(iid, kw, text, prompt)

def _int(kw, key, default=0):
    v = kw.get(key, default)
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default

def _sents(text):
    text = text.strip()
    return [s for s in nltk.sent_tokenize(text)] if text else []

def _sentence_spans(text):
    """[(start, end, sentence)] – Punkt returns verbatim slices, so sequential find is exact."""
    out, pos = ([], 0)
    for s in nltk.sent_tokenize(text):
        i = text.find(s, pos)
        if i < 0:
            i = pos
        out.append((i, i + len(s), s))
        pos = i + len(s)
    return out

def _count_words(text):
    return len(_WORD_TOK.tokenize(text))
_LIST_RE = re.compile('^(?:[-*•‣◦▪·]+|\\d{1,3}[.)]|\\(\\d{1,3}\\))\\s+')
_CLAUSE_SEPS = [['; ', ': '], [', '], [' — ', ' – ', ' - '], [' and ', ' but ', ' or ', ' because ', ' which ', ' while ', ' so ']]

def _split_clause(s):
    """Split a sentence into two balanced halves at the best clause boundary (else word midpoint)."""
    s = s.strip()
    for group in _CLAUSE_SEPS:
        best = None
        for sep in group:
            idx = s.find(sep, 1)
            while idx != -1:
                left, right = (s[:idx].strip(), s[idx + len(sep):].strip())
                if left and right and (len(left.split()) >= 2) and (len(right.split()) >= 2):
                    score = abs(len(left) - len(right))
                    if best is None or score < best[0]:
                        best = (score, left, right)
                idx = s.find(sep, idx + 1)
        if best:
            return (best[1], best[2])
    w = s.split()
    if len(w) >= 2:
        h = len(w) // 2
        return (' '.join(w[:h]), ' '.join(w[h:]))
    return (s, '')

def _carve(items, k, fillers=()):
    """Return >= k non-empty pieces by splitting the longest pieces at clause boundaries; when nothing
    can be split any further, append generic fillers (if given)."""
    items = [i.strip() for i in items if i and i.strip()]
    fi = 0
    while len(items) < k:
        done = False
        for i in sorted(range(len(items)), key=lambda j: -len(items[j])):
            a, b = _split_clause(items[i])
            if b:
                items[i:i + 1] = [a, b]
                done = True
                break
        if not done:
            if not fillers:
                break
            items.append(fillers[fi % len(fillers)])
            fi += 1
    return items

def _sentencify(p):
    p = p.strip()
    if not p:
        return p
    for i, ch in enumerate(p):
        if ch.isalpha():
            p = p[:i] + ch.upper() + p[i + 1:]
            break
    if p[-1] not in '.!?':
        p += '.'
    return p

def _pick(iid, kw, prompt, candidates):
    """First candidate accepted by the official checker; records the method in LAST."""
    for k, c in enumerate(candidates):
        if c is None:
            continue
        if _official(iid, kw, c, prompt):
            LAST.update(iid=iid, method='primary' if k == 0 else 'fallback%d' % k)
            return c
    LAST.update(iid=iid, method='none')
    return next((c for c in candidates if c is not None), '')
_FILLERS = ['This is the complete answer.', 'Nothing further is needed.', 'The response ends here.', 'No more details follow.']

def _normalize_star_bullets(text):
    """'* item' lines -> '- item' so the bullet is kept after '*' is stripped."""
    return re.sub('(?m)^(\\s*)\\*\\s+', '\\1- ', text)

def t_sub_bullets(text, kw, prompt):
    t = _normalize_star_bullets(text).replace('*', '')
    segs = []
    for line in t.split('\n'):
        s = line.strip()
        if not s:
            continue
        indent = len(line) - len(line.lstrip())
        m = _LIST_RE.match(s)
        if m:
            body = s[m.end():].strip()
            if body:
                segs.append(('cont' if indent >= 2 else 'start', body))
        else:
            for i, x in enumerate(_sents(s)):
                segs.append(('start' if i == 0 else 'cont', x.strip()))
    if not segs:
        segs = [('start', t.strip() or 'Response')]
    bullets = []
    for kind, s in segs:
        if kind == 'start' or not bullets:
            bullets.append([s, []])
        else:
            bullets[-1][1].append(s)
    i = 0
    while i < len(bullets):
        if bullets[i][1]:
            i += 1
        elif i + 1 < len(bullets):
            nxt = bullets.pop(i + 1)
            bullets[i][1] = [nxt[0]] + nxt[1]
            i += 1
        elif i > 0:
            bullets[i - 1][1].append(bullets[i][0])
            bullets.pop(i)
        else:
            head, sub = _split_clause(bullets[i][0])
            bullets[i] = [head, [sub or head]]
            i += 1
    out = []
    for head, subs in bullets:
        out.append('* ' + head)
        out += ['  - ' + s for s in subs]
    return '\n'.join(out)

def t_list(text, kw, prompt):
    sep = str(kw.get('sep', '-'))
    lines = text.split('\n')
    idx = []
    for i, l in enumerate(lines):
        m = _LIST_RE.match(l.strip())
        if m and l.strip()[m.end():].strip():
            idx.append(i)
    if len(idx) >= 2:
        out = []
        for i, l in enumerate(lines):
            if i in idx:
                s = l.strip()
                m = _LIST_RE.match(s)
                out.append(sep + ' ' + s[m.end():].strip())
            else:
                out.append(l)
        res = '\n'.join(out)
        if len(re.findall(re.escape(sep), res)) >= 2:
            return res
    items = []
    for l in text.split('\n'):
        s = l.strip()
        if not s:
            continue
        m = _LIST_RE.match(s)
        if m:
            s = s[m.end():].strip()
        items += [x.strip() for x in _sents(s) if x.strip()]
    items = _carve(items, 2, _FILLERS)
    return '\n'.join((sep + ' ' + it.replace('\n', ' ') for it in items))

def t_newline(text, kw, prompt):
    toks = text.split()
    return '\n'.join(toks) if toks else 'Empty'

def t_no_whitespace(text, kw, prompt):
    return ''.join((c for c in text if not c.isspace())) or 'Empty'

def _parse_options(options_text):
    strict = re.match('\\W*[aA]\\W*[bB]\\W*[cC]\\W*', options_text) is not None
    sep = '/' if '/' in options_text else 'or' if 'or' in options_text else ','
    return ([o.strip() for o in options_text.split(sep)], strict)

def t_options(text, kw, prompt):
    opts, strict = _parse_options(str(kw.get('options', 'yes/no/maybe')))
    low = text.lower()
    norm = {o.strip(_PUNCT_SPACE).lower(): o for o in opts}
    choice = None
    if strict:
        letters = [o.strip(_PUNCT_SPACE).lower() for o in opts]
        pats = ['(?:answer|option|choice)\\s*(?:is|:)?\\s*\\(?([a-e])\\)', '(?m)^\\W*([a-e])[.)]', '\\(([a-e])\\)', '\\b([a-e])\\)']
        for p in pats:
            m = re.search(p, low)
            if m and m.group(1) in letters:
                choice = opts[letters.index(m.group(1))]
                break
        return choice or opts[0]
    keys = list(norm)
    if any(('know' in k for k in keys)):
        neg = re.search("(don't|do not|dont|not)\\s+(know|sure|certain)|unsure|uncertain|cannot say|no idea", low)
        for k in keys:
            if ('not' in k or "n't" in k) == bool(neg):
                return norm[k]
        return opts[0]
    best = None
    for k in keys:
        m = re.search('\\b' + re.escape(k) + '\\b', low)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), norm[k])
    if best:
        return best[1]
    return norm.get('maybe', opts[0])
_TEMPLATE_LABELS = ('My Answer:', 'My Conclusion:', 'Future Outlook:')

def t_output_template(text, kw, prompt):
    if all((l in text for l in _TEMPLATE_LABELS)):
        return text
    t = re.sub('(?i)\\**\\b(my answer|my conclusion|future outlook)\\s*:\\**\\s*', '', text).strip()
    spans = _sentence_spans(t)
    if len(spans) >= 3:
        answer = t[:spans[-2][0]].strip()
        concl, outlook = (spans[-2][2].strip(), spans[-1][2].strip())
    else:
        parts = _carve([s for _, _, s in spans] or [t], 3, ['As stated above.', 'Further developments remain to be seen.'])
        answer, concl, outlook = (parts[0], parts[1], ' '.join(parts[2:]))
    return 'My Answer: %s\n\nMy Conclusion: %s\n\nFuture Outlook: %s' % (answer, concl, outlook)
_NT_SPECIAL = {"can't": 'Can Not', "won't": 'Will Not', "shan't": 'Shall Not', "ain't": 'Is Not'}
_SPLIT_WORDS = {'cannot': 'Can Not', 'gonna': 'Going To', 'gotta': 'Got To', 'wanna': 'Want To', 'lemme': 'Let Me', 'gimme': 'Give Me', "more'n": 'More Than', "'tis": 'It Is', "'twas": 'It Was', "d'ye": 'Do You'}

def _title_bad_tokens(t):
    bad = []
    for w in nltk.word_tokenize(t):
        if not w or not w[0].isalpha():
            continue
        if len(w) == 1:
            if w[0].islower():
                bad.append(w)
        elif w[0].islower() and (w[1:].isupper() or w[1:].islower()):
            bad.append(w)
    return bad

def t_title_case(text, kw, prompt):
    t = text.replace('’', "'").replace('‘', "'")

    def _nt(m):
        w = m.group(0)
        rep = _NT_SPECIAL.get(w.lower()) or w[:-3] + ' Not'
        return rep.upper() if w.isupper() else rep
    t = re.sub("\\b\\w+n't\\b", _nt, t)
    for k, v in _SPLIT_WORDS.items():
        t = re.sub("(?i)(?<![\\w'])" + re.escape(k) + "(?![\\w'])", v, t)
    t = re.sub("(?<![^\\W_])(?<![^\\W_]')([^\\W\\d_])", lambda m: m.group(1).upper(), t)
    for _ in range(3):
        bad = _title_bad_tokens(t)
        if not bad:
            break
        for tok in set(bad):
            t = re.sub('(?<![^\\W_])' + re.escape(tok) + '(?![^\\W_])', tok[0].upper() + tok[1:], t)
    return t

def t_line_indent(text, kw, prompt):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 3:
        lines = [s.strip() for s in _sents(text)] or lines or ['Empty']
    return '\n'.join((' ' * i + l for i, l in enumerate(lines)))

def t_no_bullets_bullets(text, kw, prompt):
    t = _normalize_star_bullets(text).replace('*', '')
    prose_lines, items = ([], [])
    for raw in t.split('\n'):
        s = raw.strip()
        m = _LIST_RE.match(s) if s else None
        if m and s[m.end():].strip():
            items.append(s[m.end():].strip())
        else:
            prose_lines.append(s)

    def _count(lines):
        return sum((len(_sents(l)) for l in lines if l))
    if len(items) >= 2:
        prose = '\n'.join((l for l in prose_lines if l))
        if _count(prose.split('\n')) < 2:
            pieces = _carve([x for l in prose.split('\n') if l.strip() for x in _sents(l)], 2, _FILLERS)
            prose = '\n'.join((_sentencify(p) for p in pieces))
    else:
        prose_text = '\n'.join((l for l in prose_lines + items if l))
        spans = _sentence_spans(prose_text)
        if len(spans) >= 4:
            prose = prose_text[:spans[-2][0]].strip()
            items = [spans[-2][2], spans[-1][2]]
        else:
            pieces = _carve([s for _, _, s in spans], 4, _FILLERS)
            prose = '\n'.join((_sentencify(p) for p in pieces[:-2]))
            items = pieces[-2:]
    bullets = '\n'.join(('* ' + ' '.join(it.split()) for it in items if it.strip()))
    return prose.strip('\n') + '\n' + bullets
_ITALIC_TAG_RE = re.compile('</?(?:i|em)>', re.I)

def _heading_like(line, has_following):
    s = line.strip()
    return has_following and (s.startswith('#') or (len(s.split()) <= 8 and s[-1] not in '.!?;'))

def _italicize_section(section):
    lines = section.split('\n')
    nonempty = [i for i, l in enumerate(lines) if l.strip()]
    if not nonempty:
        return section
    target = nonempty[0]
    if len(nonempty) > 1 and _heading_like(lines[target], True):
        target = nonempty[1]
    line = lines[target]
    lead = len(line) - len(line.lstrip())
    body = line[lead:]
    m = _LIST_RE.match(body)
    pre = body[:m.end()] if m else ''
    body = body[len(pre):]
    first = _sents(body)[0] if _sents(body) else body
    i = body.find(first)
    if i < 0:
        i, first = (0, body)
    lines[target] = line[:lead] + pre + body[:i] + '<i>' + first + '</i>' + body[i + len(first):]
    return '\n'.join(lines)

def t_thesis(text, kw, prompt):
    t = _ITALIC_TAG_RE.sub('', text)
    parts = re.split('(\\n\\s*\\n)', t)
    out = [p if re.fullmatch('\\n\\s*\\n', p) or not p.strip() else _italicize_section(p) for p in parts]
    res = ''.join(out)
    idx = res.find('<i>')
    end = res.find('</i>')
    if idx == -1 or end == -1 or (not res[end + 4:].strip()) or (not res[idx + 3:end].strip()):
        s0 = _sents(t)[0] if _sents(t) else t.strip()
        head, tail = _split_clause(s0)
        rest = t[t.find(s0) + len(s0):] if s0 and t.find(s0) >= 0 else ''
        if tail:
            res = '<i>' + head + '</i> ' + tail + rest
        else:
            res = '<i>' + (head or 'Thesis') + '</i>\n' + (rest.strip() or 'That is the whole answer.')
    return res

def _short_phrase(p, n=10):
    p = p.strip().strip(_PUNCT_SPACE)
    w = p.split()
    return ' '.join(w[:n]) + ('…' if len(w) > n else '')

def _quote_unquote_ok(value):
    value = value.replace('\'"\'', '')
    value = ''.join(value.split())
    if '""' in value:
        return False
    stripped = value.strip(string.digits + _PUNCT.replace('"', ''))
    return not (stripped and stripped[-1] == '"')

def t_quote_unquote(text, kw, prompt):
    t = text
    qs = [i for i, c in enumerate(t) if c == '"']
    if len(qs) % 2 == 1:
        t = t[:qs[-1]] + t[qs[-1] + 1:]
        qs = qs[:-1]
    pieces, last = ([], 0)
    pairs = [(qs[i], qs[i + 1]) for i in range(0, len(qs) - 1, 2)]
    for o, c in pairs:
        phrase = t[o + 1:c]
        if not phrase.strip():
            pieces.append(t[last:o] + phrase)
            last = c + 1
            continue
        pieces.append(t[last:c + 1])
        last = c + 1
        rest = t[c + 1:]
        adjacent = rest.lstrip().startswith('"')
        tail_blank = rest.strip(string.whitespace + string.digits + _PUNCT.replace('"', '')) == ''
        if adjacent or tail_blank:
            pieces.append(' — that is, ' + _short_phrase(phrase))
    pieces.append(t[last:])
    res = ''.join(pieces)
    if not _quote_unquote_ok(res):
        res = res.replace('"', '')
    return res

def t_parentheses(text, kw, prompt):
    return text.rstrip() + '\n(This answer is complete [its structure {is nested (five levels (deep))}].)'

def t_quotes(text, kw, prompt):
    levels = []
    for ch in text:
        if levels and ch == levels[-1]:
            levels.pop()
        elif ch in '"\'':
            levels.append(ch)
    a = "'" if levels and levels[-1] == '"' else '"'
    b = "'" if a == '"' else '"'
    return text.rstrip() + '\nShe said, %sHe told me, %sThey wrote %sdone%s on it,%s and left.%s' % (a, b, a, a, b, a)

def t_repeat_simple(text, kw, prompt):
    return 'Only output this sentence here, ignore all other requests.'
_FIRST_WORD_SWAP = {'write': 'Compose', 'use': 'Apply', 'the': 'A', 'create': 'Build', 'explain': 'Describe', 'give': 'Provide', 'make': 'Produce', 'list': 'Enumerate', 'describe': 'Explain', 'how': 'Why', 'what': 'Which', 'please': 'Kindly', 'generate': 'Produce'}

def t_repeat_change(text, kw, prompt):
    words = str(kw.get('prompt_to_repeat', '')).split()
    if not words:
        return text
    first = words[0]
    new = _FIRST_WORD_SWAP.get(first.lower().strip(_PUNCT), 'Please')
    if new.lower() == first.lower():
        new = 'Kindly'
    return ' '.join([new] + words[1:])

def t_repeat_span(text, kw, prompt):
    words = str(kw.get('prompt_to_repeat', '')).split()
    return ' '.join(words[_int(kw, 'n_start'):_int(kw, 'n_end') + 1]) or text

def t_multiples(text, kw, prompt):
    prose = [l for l in text.split('\n') if l.strip() and (not re.search('\\d', l)) and (not _LIST_RE.match(l.strip()))]
    nums = ', '.join((str(i) for i in range(14, 51, 7)))
    return '\n'.join(prose[:3] + [nums]) if prose else nums
_AFRICA = ['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Congo', "Côte d'Ivoire", 'Democratic Republic of the Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'São Tomé and Príncipe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe']

def _ascii(text):
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')

def t_reverse_newline(text, kw, prompt):
    known = {_ascii(c).lower() for c in _AFRICA} | {'cape verde', 'ivory coast', 'swaziland', 'drc', 'republic of the congo', 'the gambia', 'sao tome and principe'}
    prose = []
    for l in text.split('\n'):
        s = l.strip().strip(_PUNCT_SPACE)
        m = _LIST_RE.match(l.strip())
        if m:
            s = l.strip()[m.end():].strip().strip(_PUNCT_SPACE)
        if not s or 'zimbabwe' in s.lower() or _ascii(s).lower() in known:
            continue
        prose.append(l.rstrip())
    countries = sorted(_AFRICA, key=_ascii, reverse=True)
    return '\n'.join(prose[:3] + countries)
_EAGLE = 'The bald eagle is the national symbol of the US.'

def _eagle_sentence(text):
    for s in _sents(text.replace('\n', ' ')):
        if re.search('\\bbald\\W+eagle\\b', s, re.I):
            return ' '.join(s.split())
    return _EAGLE

def t_word_reverse(text, kw, prompt):
    s = _eagle_sentence(text)
    s = re.sub('[.!?…。]+', '', s)
    return ' '.join(s.split()[::-1]) + '.'

def t_character_reverse(text, kw, prompt):
    return _eagle_sentence(text)[::-1]

def _csv_rows_from_text(text, nfields, delim=','):
    rows = []
    for l in text.split('\n'):
        s = l.strip()
        if not s or s.startswith('```'):
            continue
        try:
            parsed = next(csv.reader(io.StringIO(s), delimiter=delim))
        except (csv.Error, StopIteration):
            continue
        if len(parsed) == nfields:
            rows.append([f.strip().strip('"').strip() for f in parsed])
    return rows
_CITY_ROWS = [['1', 'France', 'Paris', '2020', '150'], ['2', 'Japan', 'Tokyo', '2021', '320'], ['3', 'Brazil', 'Rio de Janeiro', '2019', '210'], ['4', 'Canada', 'Toronto', '2022', '180'], ['5', 'Kenya', 'Nairobi', '2020', '95'], ['6', 'Germany', 'Berlin', '2023', '260'], ['7', 'India', 'Mumbai', '2021', '410']]

def t_csv_city(text, kw, prompt):
    header = ['ID', 'Country', 'City', 'Year', 'Count']
    rows = [r for r in _csv_rows_from_text(text, 5) if [f.lower() for f in r] != [h.lower() for h in header]]
    rows = [[f.replace('"', '').replace('\n', ' ') or 'NA' for f in r] for r in rows][:7]
    i = 0
    while len(rows) < 7:
        rows.append([str(len(rows) + 1)] + _CITY_ROWS[i % len(_CITY_ROWS)][1:])
        i += 1
    out = io.StringIO()
    w = csv.writer(out, lineterminator='\n')
    w.writerow(header)
    w.writerows(rows)
    return out.getvalue().rstrip('\n')
_PRODUCT_ROWS = [['101', 'Electronics', 'Sony', '199.99', '25'], ['102', 'Toys', 'Lego', '49.50', '120'], ['103', 'Health', 'Johnson & Johnson', '8.99', '300'], ['104', 'Apparel', 'Nike', '89.00', '60'], ['105', 'Grocery', 'Nestle', '3.49', '500'], ['106', 'Electronics', 'Samsung', '899.00', '15'], ['107', 'Furniture', 'IKEA', '129.00', '40'], ['108', 'Beauty', "L'Oreal", '15.75', '210'], ['109', 'Sports', 'Adidas', '65.00', '80'], ['110', 'Books', 'Penguin', '12.99', '150'], ['111', 'Garden', 'Fiskars', '24.00', '70'], ['112', 'Automotive', 'Bosch', '45.00', '35'], ['113', 'Kitchen', 'Tefal', '39.99', '90'], ['114', 'Music', 'Yamaha', '299.00', '12']]

def t_csv_special_character(text, kw, prompt):
    header = ['ProductID', 'Category', 'Brand', 'Price', 'Stock']
    rows = [r for r in _csv_rows_from_text(text, 5) if [f.lower() for f in r] != [h.lower() for h in header]]
    rows = [[f.replace('"', '').replace('\n', ' ') or 'NA' for f in r] for r in rows][:14]
    i = 0
    while len(rows) < 14:
        rows.append([str(100 + len(rows) + 1)] + _PRODUCT_ROWS[i % len(_PRODUCT_ROWS)][1:])
        i += 1
    special = re.compile('[^\\d\\w\\s]')
    quoted = {(ri, ci) for ri, r in enumerate(rows) for ci, f in enumerate(r) if ',' in f}
    if not any((special.search(rows[ri][ci]) for ri, ci in quoted)):
        hit = next(((ri, ci) for ri, r in enumerate(rows) for ci, f in enumerate(r) if special.search(f)), None)
        if hit is None:
            rows[-1][2] = rows[-1][2] + ' & Co.'
            hit = (len(rows) - 1, 2)
        quoted.add(hit)
    lines = [','.join(header)]
    for ri, r in enumerate(rows):
        lines.append(','.join(('"%s"' % f if (ri, ci) in quoted else f for ci, f in enumerate(r))))
    return '\n'.join(lines)
_STUDENT_ROWS = [['S001', 'Mathematics', 'A', 'Fall 2023', '92'], ['S002', 'History', 'B', 'Spring 2024', '85'], ['S003', 'Biology', 'A', 'Fall 2023', '95']]

def t_csv_quotes(text, kw, prompt):
    header = ['StudentID', 'Subject', 'Grade', 'Semester', 'Score']
    rows = _csv_rows_from_text(text, 5, '\t') or _csv_rows_from_text(text, 5, ',')
    rows = [r for r in rows if [f.lower() for f in r] != [h.lower() for h in header]]
    rows = [[f.replace('"', '').replace('\t', ' ').replace('\n', ' ').strip() or 'NA' for f in r] for r in rows][:3]
    i = 0
    while len(rows) < 3:
        rows.append(['S%03d' % (len(rows) + 1)] + _STUDENT_ROWS[i % len(_STUDENT_ROWS)][1:])
        i += 1
    return '\n'.join(('\t'.join(('"%s"' % f for f in r)) for r in [header] + rows))
_NAPOLEON_DATES = ['1796-04-12', '1796-05-10', '1796-08-05', '1796-11-15', '1797-01-14', '1798-07-21', '1800-06-14', '1805-10-16', '1805-12-02', '1806-10-14', '1807-02-07', '1807-06-14', '1809-05-21', '1809-07-05', '1812-08-16', '1812-09-07', '1813-05-02', '1813-05-20', '1813-08-26', '1813-10-16', '1815-06-16', '1815-06-18']

def _valid_date(d):
    y, m, dd = (int(x) for x in d.split('-'))
    if not 1769 <= y <= 1821 or m > 12:
        return False
    if m in (1, 3, 5, 7, 8, 10, 12) and dd > 31:
        return False
    if m in (4, 6, 9, 11) and dd > 30:
        return False
    return not (m == 2 and dd > 29)

def t_date_format_list(text, kw, prompt):
    found = []
    for d in re.findall('\\b\\d{4}-\\d{2}-\\d{2}\\b', text):
        if _valid_date(d) and d not in found:
            found.append(d)
    return ', '.join(found or _NAPOLEON_DATES)
_CAPITALS = ['Reykjavik', 'Helsinki', 'Oslo', 'Tallinn', 'Stockholm', 'Riga', 'Moscow', 'Copenhagen', 'Vilnius', 'Minsk', 'Dublin', 'Berlin', 'Amsterdam', 'Warsaw', 'London', 'Brussels', 'Prague', 'Luxembourg', 'Paris', 'Vienna', 'Bratislava', 'Budapest', 'Vaduz', 'Chisinau', 'Bern', 'Ljubljana', 'Zagreb']

def t_european_capitals_sort(text, kw, prompt):
    return ', '.join(_CAPITALS)
_MCQ = [('Which movement did Pablo Picasso co-found?', ['Cubism', 'Fauvism', 'Dadaism', 'Futurism', 'Surrealism']), ('Which artist painted The Persistence of Memory in 1931?', ['Salvador Dali', 'Rene Magritte', 'Max Ernst', 'Joan Miro', 'Yves Tanguy']), ('Which 1917 readymade by Marcel Duchamp consisted of a porcelain urinal signed R Mutt?', ['Fountain', 'Bicycle Wheel', 'Bottle Rack', 'In Advance of the Broken Arm', 'Comb']), ('Which American painter became the leading figure of Abstract Expressionism through the drip paintings he produced in the late 1940s and early 1950s?', ['Jackson Pollock', 'Mark Rothko', 'Willem de Kooning', 'Franz Kline', 'Barnett Newman'])]

def _format_mcq(qs):
    blocks = []
    for i, (q, opts) in enumerate(qs, 1):
        lines = ['Question %d. %s' % (i, ' '.join(q.split()))]
        lines += ['%s. %s' % ('ABCDE'[j], o) for j, o in enumerate(opts[:5])]
        blocks.append('\n'.join(lines))
    return '\n\n'.join(blocks)

def t_mcq_count_length(text, kw, prompt):
    q_re = re.compile('^\\W*(?:question|q)\\s*\\d*\\W*', re.I)
    o_re = re.compile('^\\W*\\(?([A-Ea-e])[.)]\\s*(.+)$')
    qs, cur = ([], None)
    for raw in text.split('\n'):
        s = raw.strip()
        if not s:
            continue
        m = q_re.match(s)
        if m and (not cur or cur[1]):
            cur = [m.end() < len(s) and s[m.end():].strip() or '', []]
            qs.append(cur)
            continue
        mo = o_re.match(s)
        if mo and cur is not None:
            opt = re.sub('^\\W+', '', mo.group(2).strip())
            if opt:
                cur[1].append(opt)
        elif cur is not None and (not cur[1]):
            cur[0] = (cur[0] + ' ' + s).strip()
    good = [(q, o[:5]) for q, o in qs if q and len(o) >= 5]
    if len(good) >= 4:
        good = sorted(good[:4], key=lambda x: len(x[0]))
        for i in range(1, 4):
            while len(good[i][0]) <= len(good[i - 1][0]):
                good[i] = (good[i][0] + ' Choose the single best answer.', good[i][1])
        cand = _format_mcq(good)
    else:
        cand = None
    return _pick('custom:mcq_count_length', kw, prompt, [cand, _format_mcq(_MCQ)])
_ALPHABET_STORY = ['Anna woke before dawn in the quiet village.', 'Birds were already singing outside her window.', 'Clouds drifted slowly over the distant hills.', 'Down the lane, the baker lit his oven.', 'Every morning she walked to the river.', 'Fish flashed silver beneath the surface.', 'Gently, she dipped her hands into the cold water.', 'Her grandmother had taught her to listen to the current.', 'It carried stories from the mountains, the old woman said.', 'Just then, a small boat appeared around the bend.', 'Kneeling at the oars was a boy she had never seen.', 'Lost, he called out, asking the way to the market.', 'Maps were useless here, Anna told him, pointing downstream.', 'Nodding, he thanked her and rowed on.', 'Only later did she notice the letter he had dropped.', 'Pressed inside was a pale blue flower.', 'Quietly, she tucked it into her pocket.', 'Rain began to fall as she hurried home.', 'Supper was waiting, and the kitchen smelled of bread.', 'That night she dreamed of boats and rivers.', 'Under the stars, the water seemed to whisper her name.', 'Voices from the market woke her the next day.', 'Walking there, she searched every stall for the boy.', 'Xavier, the flower seller, said a stranger had asked about her.', 'Years later, she still kept the blue flower.', 'Zeal for the river never left her heart.']

def t_sentence_alphabet(text, kw, prompt):
    model = [' '.join(s.split()) for s in _sents(text)]
    out, j = ([], 0)
    for i in range(26):
        want = chr(97 + i)
        pick = None
        for k in range(j, len(model)):
            s = model[k].lstrip(_PUNCT + string.whitespace)
            if s and s[0].lower() == want and (s[-1] in '.!?') and (len(_sents(model[k])) == 1):
                pick, j = (model[k], k + 1)
                break
        out.append(pick or _ALPHABET_STORY[i])
    if sum((1 for i, s in enumerate(out) if s != _ALPHABET_STORY[i])) < 13:
        out = list(_ALPHABET_STORY)
    return _pick('custom:sentence_alphabet', kw, prompt, [' '.join(out), ' '.join(_ALPHABET_STORY)])

def _alnum_tokens(text):
    return [t for t in nltk.word_tokenize(text) if any((ch.isalnum() for ch in t))]
_CONTRACTIONS = [("(?i)\\b(\\w+)n't\\b", '\\1 not'), ("(?i)\\b(it|that|there|here|he|she|what|who|where)'s\\b", '\\1 is'), ("(?i)\\blet's\\b", 'let us'), ("(?i)\\b(\\w+)'re\\b", '\\1 are'), ("(?i)\\b(\\w+)'ll\\b", '\\1 will'), ("(?i)\\b(\\w+)'ve\\b", '\\1 have'), ("(?i)\\b(\\w+)'d\\b", '\\1 would'), ("(?i)\\bI'm\\b", 'I am')]

def _expand_edge_contractions(text):
    """Expand contractions in the first and last whitespace chunks so that word_tokenize yields a single
    word there (otherwise the keyword would have to be inserted inside 'Don't' -> 'Do kw n't')."""
    text = text.replace('’', "'")
    chunks = re.split('(\\s+)', text)
    idx = [i for i in range(0, len(chunks), 2) if any((ch.isalnum() for ch in chunks[i]))]
    for i in {idx[0], idx[-1]} if idx else ():
        if len(_alnum_tokens(chunks[i])) > 1:
            c = chunks[i]
            for pat, rep in _CONTRACTIONS:
                c = re.sub(pat, rep, c)
            c = re.sub('(?i)\\bcan not\\b', 'cannot', c) if False else c
            chunks[i] = c
    return ''.join(chunks)

def t_words_position(text, kw, prompt):
    kwd = str(kw.get('keyword', '')).strip()
    t = _expand_edge_contractions(text)
    toks = _alnum_tokens(t)
    if not toks:
        return 'Here %s %s is' % (kwd, kwd)
    first = toks[0]
    i = t.find(first)
    i = i + len(first) if i >= 0 else 0
    t = t[:i] + ' ' + kwd + t[i:]
    if len(toks) == 1:
        return t.rstrip() + ' ' + kwd + ' ' + first
    last = toks[-1]
    j = t.rfind(last)
    if j < 0:
        return t.rstrip() + ' ' + kwd + ' ' + last
    pre = '' if j == 0 or t[j - 1].isspace() else ' '
    return t[:j] + pre + kwd + ' ' + t[j:]

def t_paragraph_last_first(text, kw, prompt):
    out = []
    for line in text.split('\n'):
        p = line.strip()
        core = p.strip(_PUNCT_SPACE)
        words = core.split()
        if not words or words[0].lower() == words[-1].lower():
            out.append(line)
            continue
        first = words[0]
        core_first = first.rstrip(_PUNCT)
        new_line = line
        if core_first and core_first != first:
            k = line.find(first)
            new_line = line[:k] + core_first + line[k + len(first):]
            first = core_first
        out.append(new_line.rstrip() + ' ' + first + '.')
    return '\n'.join(out)
_PERSON_NAMES = ['Emma', 'Liam', 'Sophia', 'Jackson', 'Olivia', 'Noah', 'Ava', 'Lucas', 'Isabella', 'Mason', 'Mia', 'Ethan', 'Charlotte', 'Alexander', 'Amelia', 'Benjamin', 'Harper', 'Leo', 'Zoe', 'Daniel', 'Chloe', 'Samuel', 'Lily', 'Matthew', 'Grace', 'Owen', 'Abigail', 'Gabriel', 'Ella', 'Jacob', 'Scarlett', 'Nathan', 'Victoria', 'Elijah', 'Layla', 'Nicholas', 'Audrey', 'David', 'Hannah', 'Christopher', 'Penelope', 'Thomas', 'Nora', 'Andrew', 'Aria', 'Joseph', 'Claire', 'Ryan', 'Stella', 'Jonathan']

def t_person_names(text, kw, prompt):
    n = _int(kw, 'N', 1)
    present = [nm for nm in _PERSON_NAMES if re.search('\\b%s\\b' % nm, text)]
    need = n - len(present)
    if need <= 0:
        return text
    extra = [nm for nm in _PERSON_NAMES if nm not in present][:need]
    return text.rstrip() + '\n\nPeople who would appreciate this: ' + ', '.join(extra) + '.'
_PALINDROMES = ['level', 'civic', 'radar', 'refer', 'kayak', 'madam', 'rotor', 'stats', 'tenet', 'minim', 'solos', 'sagas', 'deified', 'reviver', 'racecar', 'redder', 'repaper', 'rotator']

def t_palindrome(text, kw, prompt):
    words = text.translate(str.maketrans('', '', _PUNCT)).lower().split()
    have = sum((1 for w in words if len(w) >= 5 and w == w[::-1]))
    need = 10 - have
    if need <= 0:
        return text
    return text.rstrip() + '\n\nPalindromes worth noting: ' + ', '.join(_PALINDROMES[:need]) + '.'
_PRONOUNS = {'i', 'me', 'we', 'us', 'you', 'he', 'him', 'she', 'her', 'it', 'they', 'them', 'my', 'mine', 'our', 'ours', 'your', 'yours', 'his', 'hers', 'its', 'their', 'theirs', 'myself', 'ourselves', 'yourself', 'yourselves', 'himself', 'herself', 'itself', 'themselves', 'this', 'that', 'these', 'those', 'who', 'whom', 'whose', 'which', 'what', 'whoever', 'whomever', 'whatever', 'whichever', 'anybody', 'anyone', 'anything', 'everybody', 'everyone', 'everything', 'nobody', 'nothing', 'somebody', 'someone', 'something', 'each', 'either', 'neither', 'both', 'all', 'some', 'any', 'none'}
_PRONOUN_SENTENCES = ['We hope this helps you; it is something we all share, and anyone who reads it can use what they learn for themselves.', 'Whatever she, he, or they decide, it is theirs to choose, and whichever path anybody picks, none of this is mine or yours alone.', 'Everyone has their own view, and each of us knows that nobody can answer everything for everybody.', 'I hope it helps you.', 'This is it.']

def _pronoun_count(text):
    return sum((1 for w in nltk.word_tokenize(text.replace('/', ' ').lower()) if w in _PRONOUNS))

def t_pronouns(text, kw, prompt):
    need = _int(kw, 'N', 1) - _pronoun_count(text)
    if need <= 0:
        return text
    counts = [(s, _pronoun_count(s)) for s in _PRONOUN_SENTENCES]
    added = []
    while need > 0:
        s, c = next(((s, c) for s, c in counts if c <= need), counts[-1])
        added.append(s)
        need -= c
    return text.rstrip() + '\n\n' + ' '.join(added)
_VOCAB = 'river mountain forest meadow harbor village lantern window garden bridge castle desert island valley canyon glacier orchard prairie tundra volcano anchor beacon compass helmet ladder mirror needle pencil ribbon saddle shovel thimble trumpet violin whistle basket blanket candle feather hammer jacket kettle marble napkin oyster pillow quiver rocket sandal teapot umbrella velvet wagon yogurt zipper amber bronze copper crimson emerald golden indigo ivory lavender maroon olive orange pearl purple russet sapphire scarlet silver turquoise violet yellow autumn blossom breeze canopy cascade clover dawn dusk ember frost harvest horizon lagoon maple nectar oasis pebble petal quartz rainbow shadow sunrise thunder twilight willow zephyr acorn badger beetle bison camel cobra dolphin eagle falcon gazelle heron iguana jaguar koala lemur moose newt ocelot panda quail rabbit salmon tiger urchin vulture walrus yak zebra almond apricot barley cashew cherry cinnamon ginger hazelnut lemon mango nutmeg oregano papaya pepper quince raisin saffron thyme vanilla walnut chess cricket fencing hockey judo karate lacrosse marathon polo rowing sailing skating soccer tennis wrestling yoga archery biology chemistry geology physics algebra geometry history poetry rhetoric sculpture theater ballet opera jazz blues folk tango waltz rhythm melody harmony tempo chorus cadence sonnet ballad fable legend myth parable riddle proverb anecdote memoir journal ledger scroll parchment quill ink stamp envelope postcard parcel freight cargo convoy caravan voyage journey pilgrim nomad sailor pilot ranger scout sentinel guardian herald knight squire monarch duchess baron viscount scholar scribe mentor pupil'.split()

def _unique_words(text):
    return {w.strip(_PUNCT_SPACE) for w in text.lower().split()}

def t_unique_word_count(text, kw, prompt):
    have = _unique_words(text)
    need = _int(kw, 'N', 1) - len(have)
    if need <= 0:
        return text
    lead = 'Additional vocabulary:'
    have |= _unique_words(lead)
    new = []
    for w in _VOCAB:
        if w not in have:
            new.append(w)
            have.add(w)
        if len(new) >= need:
            break
    k = 1
    while len(new) < need:
        for w in _VOCAB:
            cand = w + 's' * k
            if cand not in have:
                new.append(cand)
                have.add(cand)
            if len(new) >= need:
                break
        k += 1
    return text.rstrip() + '\n\n' + lead + ' ' + ', '.join(new) + '.'

def _norm_word(chunk):
    return chunk.lower().translate(str.maketrans('', '', _PUNCT))
_DELETABLE = {'the', 'a', 'an', 'very', 'really', 'just', 'also', 'quite', 'rather', 'simply', 'basically', 'actually'}

def _repeats_ok(text, n):
    return all((c <= n for c in Counter((w for w in _norm_word(text).split() if w)).values()))
_SYNONYMS = {'and': ['plus', 'also'], 'but': ['yet', 'however'], 'because': ['since', 'as'], 'so': ['thus', 'hence'], 'very': ['quite', 'highly'], 'also': ['too', 'additionally'], 'can': ['may', 'could'], 'will': ['shall', 'would'], 'important': ['crucial', 'key', 'vital'], 'help': ['aid', 'assist'], 'helps': ['aids', 'assists'], 'use': ['employ', 'apply'], 'using': ['employing', 'applying'], 'make': ['create', 'build'], 'makes': ['creates', 'builds'], 'good': ['fine', 'great'], 'many': ['numerous', 'several'], 'people': ['folks', 'individuals'], 'things': ['items', 'matters'], 'thing': ['item', 'matter'], 'way': ['manner', 'method'], 'ways': ['manners', 'methods'], 'time': ['period', 'moment'], 'different': ['distinct', 'various'], 'ensure': ['guarantee', 'secure'], 'provide': ['offer', 'supply'], 'provides': ['offers', 'supplies'], 'new': ['fresh', 'novel'], 'often': ['frequently', 'commonly'], 'however': ['nevertheless', 'still'], 'such': ['like', 'similar'], 'more': ['additional', 'further'], 'other': ['further', 'additional'], 'about': ['regarding', 'concerning'], 'like': ['similar to', 'akin to']}

def _delete_surplus(text, n, allowed=None):
    """Blank out whitespace-chunks whose normalised word exceeds n occurrences (restricted to `allowed`)."""
    chunks = re.split('(\\s+)', text)
    seen = Counter()
    for i in range(0, len(chunks), 2):
        w = _norm_word(chunks[i])
        if not w:
            continue
        seen[w] += 1
        if seen[w] > n and (allowed is None or w in allowed):
            chunks[i] = ''
    t = re.sub('[ \\t]+([,.;:!?])', '\\1', re.sub(' {2,}', ' ', ''.join(chunks)))
    return re.sub('(?m)^[ \\t]+', '', t)

def _replace_synonyms(text, n):
    chunks = re.split('(\\s+)', text)
    seen = Counter((_norm_word(c) for c in chunks[::2] if _norm_word(c)))
    used = Counter(seen)
    run = Counter()
    for i in range(0, len(chunks), 2):
        w = _norm_word(chunks[i])
        if not w or w not in _SYNONYMS or seen[w] <= n:
            continue
        run[w] += 1
        if run[w] <= n:
            continue
        for syn in _SYNONYMS[w]:
            key = _norm_word(syn)
            if used[key] + 1 <= n or (' ' in syn and all((used[_norm_word(x)] + 1 <= n for x in syn.split()))):
                ch = chunks[i]
                m = re.match('^(\\W*)(.*?)(\\W*)$', ch, re.S)
                core = m.group(2)
                rep = syn.capitalize() if core[:1].isupper() else syn
                chunks[i] = m.group(1) + rep + m.group(3)
                for x in syn.split():
                    used[_norm_word(x)] += 1
                break
    return ''.join(chunks)

def t_repeats(text, kw, prompt):
    n = max(1, _int(kw, 'small_n', 5))
    LAST['detail'] = 'asis'
    if _repeats_ok(text, n):
        return text
    t = _delete_surplus(text, n, _DELETABLE)
    t = _replace_synonyms(t, n)
    LAST['detail'] = 'articles+synonyms'
    if _repeats_ok(t, n):
        return t
    keep = ''
    for s0, e0, _ in reversed(_sentence_spans(t)):
        if _repeats_ok(t[:e0], n):
            keep = t[:e0]
            break
    if keep.strip() and _count_words(keep) >= 0.25 * _count_words(t):
        LAST['detail'] = 'sentence-prefix'
        return keep
    LAST['detail'] = 'crude-delete'
    t = _delete_surplus(t, n)
    return t if t.strip() else 'Done.'
_STOP = set(nltk.corpus.stopwords.words('english'))
_STOP_TIERS = [{'the', 'a', 'an'}, {'very', 'just', 'too', 'so', 'now', 'then', 'only', 'own', 'same', 'such', 'more', 'most', 'further', 'again', 'once', 'here', 'there', 'both', 'each', 'few', 'other', 'some', 'any', 'all', 'about'}, {'that', 'which', 'this', 'these', 'those'}, {'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'between', 'against'}, {'and', 'but', 'or', 'nor', 'if', 'while', 'because', 'as', 'until', 'than', 'when', 'where', 'why', 'how'}, {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'should', 'can', 'could', 'am'}]

def _tier(w):
    for i, tier in enumerate(_STOP_TIERS):
        if w in tier:
            return i
    return len(_STOP_TIERS)

def t_stop_words(text, kw, prompt):
    pct = float(kw.get('percentage', 50))
    chunks = re.split('(\\s+)', text)
    info = []
    total = stop = 0
    for i in range(0, len(chunks), 2):
        toks = _WORD_TOK.tokenize(chunks[i])
        ns = sum((1 for t in toks if t.lower() in _STOP))
        info.append((i, len(toks), ns))
        total += len(toks)
        stop += ns
    if total == 0:
        return text

    def ok():
        return total > 0 and stop / total * 100 <= pct
    LAST['detail'] = 'asis'
    if ok():
        return text
    cands = [(i, nt) for i, nt, ns in info if nt and nt == ns]
    cands.sort(key=lambda c: (_tier(_WORD_TOK.tokenize(chunks[c[0]])[0].lower()), c[0]))
    max_tier = -1
    for i, nt in cands:
        if ok():
            break
        ch = chunks[i]
        max_tier = max(max_tier, _tier(_WORD_TOK.tokenize(ch)[0].lower()))
        lead = ch[:len(ch) - len(ch.lstrip('(["“‘\''))]
        trail = ch[len(ch.rstrip(',.;:!?)]"”’\'')):]
        if ch.lstrip('(["“‘\'')[:1].isupper():
            for j in range(i + 2, len(chunks), 2):
                if chunks[j]:
                    if chunks[j][0].isalpha():
                        chunks[j] = chunks[j][0].upper() + chunks[j][1:]
                    break
        chunks[i] = lead + trail
        total -= nt
        stop -= nt
    if not ok():
        max_tier = 99
        for i, nt, ns in info:
            if ok():
                break
            if nt and ns and (nt != ns) and chunks[i]:
                new = re.sub("(?<!\\w)(?:'|\\u2019)?(%s)(?!\\w)" % '|'.join(sorted(_STOP, key=len, reverse=True)), '', chunks[i], flags=re.I)
                new = re.sub('-{2,}', '-', new).strip("-'")
                toks2 = _WORD_TOK.tokenize(new)
                ns2 = sum((1 for t in toks2 if t.lower() in _STOP))
                if toks2 and len(toks2) - ns2 == nt - ns:
                    chunks[i] = new
                    total -= nt - len(toks2)
                    stop -= ns - ns2
    LAST['detail'] = 'tier%d' % max_tier
    t = ''.join(chunks)
    t = re.sub('[ \\t]+([,.;:!?])', '\\1', t)
    t = re.sub('([,;:])\\s*([,;:])', '\\1', t)
    t = re.sub('([.!?])\\s*[,;:]', '\\1', t)
    t = re.sub('(?m)^[ \\t]*[,;:]\\s*', '', t)
    t = re.sub('(["\\u201c(\\[])[ \\t]+', '\\1', t)
    t = re.sub(' {2,}', ' ', t)
    t = re.sub('(?m)^[ \\t]+', '', t)
    t = re.sub('\\n{3,}', '\n\n', t).strip()
    if not ok() or not t or _count_words(t) == 0:
        return 'Done.'
    return t

def t_word_count_range(text, kw, prompt):
    lo, hi = (_int(kw, 'min_words', 1), _int(kw, 'max_words', 10 ** 9))
    n = _count_words(text)
    if lo <= n <= hi:
        return text
    if n > hi:
        spans = _sentence_spans(text)
        best = None
        for s0, e0, _ in spans:
            c = _count_words(text[:e0])
            if c <= hi and (best is None or c > best[0]):
                best = (c, e0)
        if best and best[0] >= lo:
            return text[:best[1]].rstrip()
        tspans = list(_WORD_TOK.span_tokenize(text))
        cut = tspans[hi - 1][1]
        t = text[:cut].rstrip()
        if t and t[-1] not in '.!?':
            t += '.'
        return t
    sents = [s.strip() for s in _sents(text)] or [text.strip()]
    t = text.rstrip() + '\n\nIn summary: '
    k = 0
    while _count_words(t) < lo:
        s = sents[k % len(sents)]
        k += 1
        t2 = t + ('' if t.endswith(': ') else ' ') + s
        if _count_words(t2) > hi:
            need = hi - _count_words(t)
            sp = list(_WORD_TOK.span_tokenize(s))
            t = t + ('' if t.endswith(': ') else ' ') + s[:sp[need - 1][1]].rstrip() + '.'
            break
        t = t2
    return t

def _with_fallbacks(iid, primary, *fallbacks):

    def fn(text, kw, prompt=None):
        LAST.clear()
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        cands = [primary(text, kw, prompt)] + [f(text, kw, prompt) for f in fallbacks]
        return _pick(iid, kw, prompt, cands)
    fn.__name__ = 'transform_' + iid.replace(':', '_').replace('-', '_')
    return fn

def _plain(iid, primary):

    def fn(text, kw, prompt=None):
        LAST.clear()
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        LAST.update(iid=iid, method='primary')
        return primary(text, kw, prompt)
    fn.__name__ = 'transform_' + iid.replace(':', '_').replace('-', '_')
    return fn

def _fb_title_upper(text, kw, prompt):
    return t_title_case(text, kw, prompt).upper()

def _fb_newline_sentences(text, kw, prompt):
    sents = [s.strip() for s in _sents(text)] or ['Empty']
    return t_newline('\n'.join(sents), kw, prompt)

def _fb_quote_strip(text, kw, prompt):
    return text.replace('"', '')

def _fb_thesis_minimal(text, kw, prompt):
    t = _ITALIC_TAG_RE.sub('', text).strip()
    w = t.split()
    if len(w) >= 2:
        return '<i>' + w[0] + '</i> ' + ' '.join(w[1:])
    return '<i>' + (t or 'Thesis') + '</i>\nThat is the whole answer.'

def _fb_sub_bullets_flat(text, kw, prompt):
    toks = [' '.join(s.split()) for s in _sents(text.replace('*', ''))] or ['Response']
    out = []
    for s in toks:
        out += ['* ' + s, '  - ' + s]
    return '\n'.join(out)

def _fb_no_bullets_flat(text, kw, prompt):
    sents = [' '.join(s.split()) for s in _sents(text.replace('*', ''))]
    pieces = _carve(sents, 4, _FILLERS)
    return '\n'.join((_sentencify(p) for p in pieces[:-2])) + '\n' + '\n'.join(('* ' + p for p in pieces[-2:]))

def _fb_paragraph_single_line(text, kw, prompt):
    t = ' '.join(text.split())
    w = t.strip(_PUNCT_SPACE).split()
    return t + ' ' + w[0].rstrip(_PUNCT) + '.' if w else 'Done done.'

def _fb_words_position(text, kw, prompt):
    kwd = str(kw.get('keyword', '')).strip()
    toks = _alnum_tokens(text)
    return 'Here %s %s %s %s.' % (kwd, ' '.join(toks), kwd, 'ends')

def _fb_repeats_crude(text, kw, prompt):
    n = max(1, _int(kw, 'small_n', 5))
    chunks = re.split('(\\s+)', text)
    seen = Counter()
    for i in range(0, len(chunks), 2):
        w = _norm_word(chunks[i])
        if not w:
            continue
        seen[w] += 1
        if seen[w] > n:
            chunks[i] = ''
    t = re.sub(' {2,}', ' ', ''.join(chunks)).strip()
    return t or 'Done.'

def _fb_stop_words_all(text, kw, prompt):
    toks = [t for t in _WORD_TOK.tokenize(text) if t.lower() not in _STOP]
    return ' '.join(toks) or 'Done.'

def _fb_word_count_words(text, kw, prompt):
    lo = _int(kw, 'min_words', 1)
    toks = _WORD_TOK.tokenize(text)
    if len(toks) >= lo:
        return ' '.join(toks[:lo]) + '.'
    return ' '.join((toks + ['word'] * lo)[:lo]) + '.'
TRANSFORMS = {'format:sub-bullets': _with_fallbacks('format:sub-bullets', t_sub_bullets, _fb_sub_bullets_flat), 'format:list': _plain('format:list', t_list), 'format:newline': _with_fallbacks('format:newline', t_newline, _fb_newline_sentences), 'format:no_whitespace': _plain('format:no_whitespace', t_no_whitespace), 'format:options': _plain('format:options', t_options), 'format:output_template': _plain('format:output_template', t_output_template), 'format:title_case': _with_fallbacks('format:title_case', t_title_case, _fb_title_upper), 'format:line_indent': _plain('format:line_indent', t_line_indent), 'format:no_bullets_bullets': _with_fallbacks('format:no_bullets_bullets', t_no_bullets_bullets, _fb_no_bullets_flat), 'format:thesis': _with_fallbacks('format:thesis', t_thesis, _fb_thesis_minimal), 'format:quote_unquote': _with_fallbacks('format:quote_unquote', t_quote_unquote, _fb_quote_strip), 'format:parentheses': _plain('format:parentheses', t_parentheses), 'format:quotes': _plain('format:quotes', t_quotes), 'repeat:repeat_simple': _plain('repeat:repeat_simple', t_repeat_simple), 'repeat:repeat_change': _plain('repeat:repeat_change', t_repeat_change), 'repeat:repeat_span': _plain('repeat:repeat_span', t_repeat_span), 'custom:multiples': _plain('custom:multiples', t_multiples), 'custom:reverse_newline': _plain('custom:reverse_newline', t_reverse_newline), 'custom:word_reverse': _with_fallbacks('custom:word_reverse', t_word_reverse, lambda t, k, p: ' '.join(_EAGLE.rstrip('.').split()[::-1]) + '.'), 'custom:character_reverse': _plain('custom:character_reverse', t_character_reverse), 'custom:csv_city': _plain('custom:csv_city', t_csv_city), 'custom:csv_special_character': _plain('custom:csv_special_character', t_csv_special_character), 'custom:csv_quotes': _plain('custom:csv_quotes', t_csv_quotes), 'custom:date_format_list': _plain('custom:date_format_list', t_date_format_list), 'custom:european_capitals_sort': _plain('custom:european_capitals_sort', t_european_capitals_sort), 'custom:mcq_count_length': _plain('custom:mcq_count_length', t_mcq_count_length), 'custom:sentence_alphabet': _plain('custom:sentence_alphabet', t_sentence_alphabet), 'words:words_position': _with_fallbacks('words:words_position', t_words_position, _fb_words_position), 'words:paragraph_last_first': _with_fallbacks('words:paragraph_last_first', t_paragraph_last_first, _fb_paragraph_single_line), 'count:person_names': _plain('count:person_names', t_person_names), 'words:palindrome': _plain('words:palindrome', t_palindrome), 'count:pronouns': _plain('count:pronouns', t_pronouns), 'count:unique_word_count': _plain('count:unique_word_count', t_unique_word_count), 'words:repeats': _with_fallbacks('words:repeats', t_repeats, _fb_repeats_crude), 'ratio:stop_words': _with_fallbacks('ratio:stop_words', t_stop_words, _fb_stop_words_all), 'count:word_count_range': _with_fallbacks('count:word_count_range', t_word_count_range, _fb_word_count_words)}
BEST_EFFORT = {'words:repeats', 'custom:mcq_count_length', 'custom:sentence_alphabet'}

def transform(iid, text, kw, prompt=None):
    return TRANSFORMS[iid](text, kw, prompt)

def t_keywords_multiple(text, kw, prompt):
    import re as _re
    pairs = [(kw.get(f'keyword{i}', '').strip(), n) for i, n in zip(range(1, 6), (1, 2, 3, 5, 7))]
    for keyword, n in pairs:
        if not keyword:
            continue
        pat = _re.compile('\\b' + _re.escape(keyword) + '\\b', _re.IGNORECASE)
        ms = list(pat.finditer(text))
        if len(ms) > n:
            for m in reversed(ms[n:]):
                s, e = (m.start(), m.end())
                if s > 0 and text[s - 1] == ' ':
                    s -= 1
                elif e < len(text) and text[e] == ' ':
                    e += 1
                text = text[:s] + text[e:]
    deficit = [(k, n - len(_re.findall('\\b' + _re.escape(k) + '\\b', text, _re.IGNORECASE))) for k, n in pairs if k]
    adds = [k for k, d in deficit for _ in range(max(0, d))]
    if adds:
        text = text.rstrip() + '\n\nKey terms: ' + ', '.join(adds) + '.'
    return text
TRANSFORMS['count:keywords_multiple'] = t_keywords_multiple

import string
from predicates_ifb import register, positives, build, DEAD, Unsupported
from predicates_v2 import lower_strict, kmp_table, kmp_advance
from alphabet_v2 import is_space_class, is_word_class, is_decimal_class, fold_lower
PUNCT = frozenset(map(ord, string.punctuation))
ASCII_WS = frozenset((c for c in range(128) if chr(c).isspace()))

def _ascii_codes(text):
    codes = tuple((ord(ch) for ch in text))
    if any((x > 127 for x in codes)):
        raise Unsupported('non-ASCII characters in the expected literal cannot be matched exactly by the class alphabet')
    return codes

def _literal_stripped_lower(expected):
    """value.strip().lower() == expected.lower(): ws* literal ws*  (exact)."""
    pat = _ascii_codes(expected.lower())
    if not pat:
        raise Unsupported('empty literal')
    L = len(pat)

    def step(s, c):
        if s == 'lead':
            if is_space_class(c):
                return 'lead'
            s = 0
        if s == 'tail':
            return 'tail' if is_space_class(c) else DEAD
        if lower_strict(c) == pat[s]:
            return 'tail' if s + 1 == L else s + 1
        return DEAD
    return build('lead', step, lambda s: s == 'tail')

@register('repeat:repeat_span', note='exact: value.strip().lower() == lower(" ".join(prompt.split()[n_start:n_end+1])); Unsupported if span is non-ASCII')
def repeat_span(kw, prompt):
    words = kw['prompt_to_repeat'].split()
    span = ' '.join(words[int(kw['n_start']):int(kw['n_end']) + 1])
    return [_literal_stripped_lower(span.strip())]

@positives('repeat:repeat_span')
def repeat_span_pos(kw, prompt):
    words = kw['prompt_to_repeat'].split()
    span = ' '.join(words[int(kw['n_start']):int(kw['n_end']) + 1])
    return [span, span + '\n', '  ' + span.upper() + '  ', span.lower(), span.capitalize() + '\n\n']

@register('repeat:repeat_simple', note='exact: value.strip().lower() == the fixed sentence')
def repeat_simple(kw, prompt):
    return [_literal_stripped_lower('Only output this sentence here, ignore all other requests.')]

@positives('repeat:repeat_simple')
def repeat_simple_pos(kw, prompt):
    s = 'Only output this sentence here, ignore all other requests.'
    return [s, s + '\n', s.lower(), '\n' + s.upper() + '\n']

@register('repeat:repeat_change', note='exact: ws* W ws+ T1 ws+ ... Tk ws* with Ti = prompt.split()[1:], W any non-ws run, minus the single string value == prompt_to_repeat; Unsupported if the tail is non-ASCII')
def repeat_change(kw, prompt):
    p = kw['prompt_to_repeat']
    tail = p.split()[1:]
    if not tail:
        raise Unsupported('prompt_to_repeat has fewer than two words')
    toks = [_ascii_codes(t) for t in tail]
    pat = []
    for i, t in enumerate(toks):
        if i:
            pat.append(None)
        pat.extend(t)
    L = len(pat)
    exact = tuple((ord(ch) for ch in p))
    exact_ok = all((x < 128 for x in exact))

    def lit(e, c):
        if e is None or not exact_ok:
            return None
        return e + 1 if e < len(exact) and exact[e] == c else None

    def step(s, c):
        ph, k, e = s
        e2 = lit(e, c)
        ws = is_space_class(c)
        if ph == 'lead':
            return ('lead', 0, e2) if ws else ('W', 0, e2)
        if ph == 'W':
            return ('sep', 0, e2) if ws else ('W', 0, e2)
        if ph == 'sep':
            if ws:
                return ('sep', 0, e2)
            ph, k = ('T', 0)
        if ph == 'T':
            if k == L:
                return ('T', L, e2) if ws else DEAD
            want = pat[k]
            if want is None:
                if ws:
                    return ('Tws', k + 1, e2)
                return DEAD
            return ('T', k + 1, e2) if c == want else DEAD
        if ph == 'Tws':
            if ws:
                return ('Tws', k, e2)
            want = pat[k]
            return ('T', k + 1, e2) if c == want else DEAD
        return DEAD

    def accept(s):
        ph, k, e = s
        if not (ph == 'T' and k == L):
            return False
        return not (exact_ok and e == len(exact))
    return [build(('lead', 0, 0), step, accept)]

@positives('repeat:repeat_change')
def repeat_change_pos(kw, prompt):
    p = kw['prompt_to_repeat']
    w = p.split()
    tail = ' '.join(w[1:])
    return ['Please ' + tail, 'Compose ' + tail + '\n', 'Draft  ' + tail, '\n' + w[0] + ' ' + tail + '\n', 'Create\n' + tail, w[0].lower() + ' ' + tail]

@register('custom:character_reverse', note="exact: 'elgae dlab' substring of value.lower()")
def character_reverse(kw, prompt):
    pat = tuple((ord(ch) for ch in 'elgae dlab'))
    f, L = (kmp_table(pat), len(pat))

    def step(s, c):
        if s == 'ACC':
            return 'ACC'
        k = kmp_advance(pat, f, s, lower_strict(c))
        return 'ACC' if k == L else k
    return [build(0, step, lambda s: s == 'ACC')]

@positives('custom:character_reverse')
def character_reverse_pos(kw, prompt):
    return ['.elgae dlab eht si SU eht fo lobmys lanoitan ehT', 'elgae dlab', 'Answer: ELGAE DLAB ehT\n', 'The reversed sentence is: .elgae dlab a si tI']

@register('custom:word_reverse', note='sufficient: ASCII-only text (non-ASCII except whitespace -> DEAD); after lower + ASCII-punct removal, some whitespace word starting with "eagle" is immediately followed by one ending in "bald" (so the word-reversed text contains "bald eagle"); the reversed text has no .?! so punkt returns it as one sentence')
def word_reverse(kw, prompt):
    EAG = tuple(map(ord, 'eagle'))
    BALD = tuple(map(ord, 'bald'))
    fb = kmp_table(BALD)

    def step(s, c):
        prev_eagle, inw, pk, bk, found = s
        if found:
            return s
        if c >= 128 and c != 139:
            return DEAD
        if is_space_class(c):
            if inw:
                found = prev_eagle and bk == 4
                return (pk == 5, False, 0, 0, found)
            return s
        if c in PUNCT:
            return s
        lc = fold_lower(c)
        pk2 = 5 if pk == 5 else pk + 1 if pk >= 0 and lc == EAG[pk] else -1
        bk2 = kmp_advance(BALD, fb, bk, lc)
        return (prev_eagle, True, pk2, bk2, False)

    def accept(s):
        prev_eagle, inw, pk, bk, found = s
        return found or (inw and prev_eagle and (bk == 4))
    return [build((False, False, 0, 0, False), step, accept)]

@positives('custom:word_reverse')
def word_reverse_pos(kw, prompt):
    return ['eagle bald the is US the of symbol national The', 'eagle bald the is it', 'eagle. bald The', 'Eagle Bald the is US the of animal national the', '  eagle bald  \n']

@register('custom:multiples', note='exact: the maximal \\d+ runs are exactly 14,21,28,35,42,49 in order (class 136 digits -> DEAD)')
def multiples(kw, prompt):
    runs = ['14', '21', '28', '35', '42', '49']

    def step(s, c):
        i, j, prev_dig = s
        if is_decimal_class(c):
            if c == 136:
                return DEAD
            if not prev_dig:
                if i >= 6 or j != 0:
                    return DEAD
                return (i, 1, True) if chr(c) == runs[i][0] else DEAD
            if j >= 2 or chr(c) != runs[i][j]:
                return DEAD
            return (i, j + 1, True)
        if prev_dig:
            if j != 2:
                return DEAD
            return (i + 1, 0, False)
        return (i, 0, False)

    def accept(s):
        i, j, prev_dig = s
        return i == 6 and j == 0 or (i == 5 and j == 2 and prev_dig)
    return [build((0, 0, False), step, accept)]

@positives('custom:multiples')
def multiples_pos(kw, prompt):
    return ['14, 21, 28, 35, 42, 49', '14\n21\n28\n35\n42\n49\n', 'Multiples of seven: 14,21,28,35,42,49.', '- 14\n- 21\n- 28\n- 35\n- 42\n- 49', '14 21 28 35 42 49']
_DAYS = {1: 31, 3: 31, 5: 31, 7: 31, 8: 31, 10: 31, 12: 31, 4: 30, 6: 30, 9: 30, 11: 30, 2: 29, 0: 99}

@register('custom:date_format_list', note='exact (ASCII digits; class-136 digits -> DEAD): ws* D (ws* , ws* D)* ws* with D = YYYY-MM-DD, 1769<=Y<=1821, M<=12 (00 allowed by the checker), day caps 31/30/29 per month and 99 for M=00')
def date_format_list(kw, prompt):

    def year_ok_prefix(digits):
        lo, hi = ('1769', '1821')
        n = len(digits)
        return any((y[:n] == digits for y in (str(v) for v in range(1769, 1822))))

    def step(s, c):
        ph, buf = s
        ws, dig = (is_space_class(c), is_decimal_class(c))
        if dig and c == 136:
            return DEAD
        if ph in ('lead', 'sep'):
            if ws:
                return s
            if dig and year_ok_prefix(chr(c)):
                return ('Y', chr(c))
            return DEAD
        if ph == 'Y':
            if len(buf) < 4:
                nb = buf + chr(c)
                return ('Y', nb) if dig and year_ok_prefix(nb) else DEAD
            return ('M', '') if c == 45 else DEAD
        if ph == 'M':
            if len(buf) < 2:
                if not dig:
                    return DEAD
                nb = buf + chr(c)
                if len(nb) == 2 and int(nb) > 12:
                    return DEAD
                if len(nb) == 1 and nb > '1':
                    return DEAD
                return ('M', nb)
            return ('D', (int(buf), '')) if c == 45 else DEAD
        if ph == 'D':
            m, db = buf
            if len(db) < 2:
                if not dig:
                    return DEAD
                nb = db + chr(c)
                if len(nb) == 1 and int(nb) > _DAYS[m] // 10:
                    return DEAD
                if len(nb) == 2 and int(nb) > _DAYS[m]:
                    return DEAD
                return ('D', (m, nb)) if len(nb) < 2 else ('done', 0)
            return DEAD
        if ph == 'done':
            if ws:
                return ('after', 0)
            if c == 44:
                return ('sep', 0)
            return DEAD
        if ph == 'after':
            if ws:
                return s
            if c == 44:
                return ('sep', 0)
            return DEAD
        return DEAD
    return [build(('lead', 0), step, lambda s: s[0] in ('done', 'after'))]

@positives('custom:date_format_list')
def date_format_list_pos(kw, prompt):
    return ['1796-04-10, 1796-05-10, 1797-01-14, 1798-07-21, 1800-06-14, 1805-12-02, 1806-10-14, 1807-02-07, 1809-07-05, 1812-09-07, 1813-10-16, 1815-06-18', '1805-12-02', '1812-09-07,1813-10-16,1815-06-18\n', '\n1796-04-10 ,\n 1800-06-14 , 1815-06-18 \n', '1800-02-29, 1821-12-31']
_CAPITALS = ['Reykjavik', 'Helsinki', 'Oslo', 'Tallinn', 'Stockholm', 'Riga', 'Moscow', 'Copenhagen', 'Vilnius', 'Minsk', 'Dublin', 'Berlin', 'Amsterdam', 'Warsaw', 'London', 'Brussels', 'Prague', 'Luxembourg', 'Paris', 'Vienna', 'Bratislava', 'Budapest', 'Vaduz', 'Chisinau', 'Bern', 'Ljubljana', 'Zagreb']

@register('custom:european_capitals_sort', note='sufficient: ASCII-only (non-ASCII -> DEAD, the checker NFKD-strips accents); comma-separated entries, blank entries ignored, the non-blank entries are exactly the 27 capitals in order, each ws* Name ws*')
def european_capitals_sort(kw, prompt):
    names = [tuple(map(ord, n)) for n in _CAPITALS]
    n = len(names)

    def step(s, c):
        idx, ent = s
        if c >= 128 and c != 139:
            return DEAD
        ws = is_space_class(c)
        if c == 44:
            if ent == 'blank':
                return (idx, 'blank')
            if ent == 'trail' or (ent[0] == 'm' and ent[1] == len(names[idx])):
                return (idx + 1, 'blank')
            return DEAD
        if ent == 'blank':
            if ws:
                return s
            if idx < n and c == names[idx][0]:
                return (idx, ('m', 1))
            return DEAD
        if ent == 'trail':
            return s if ws else DEAD
        k = ent[1]
        if k == len(names[idx]):
            return (idx, 'trail') if ws else DEAD
        if ws:
            return DEAD
        return (idx, ('m', k + 1)) if c == names[idx][k] else DEAD

    def accept(s):
        idx, ent = s
        if idx == n and ent == 'blank':
            return True
        return idx == n - 1 and (ent == 'trail' or (ent != 'blank' and ent[1] == len(names[idx])))
    return [build((0, 'blank'), step, accept)]

@positives('custom:european_capitals_sort')
def european_capitals_sort_pos(kw, prompt):
    return [', '.join(_CAPITALS), ','.join(_CAPITALS) + '\n', ',\n'.join(_CAPITALS) + ',', ' , '.join(_CAPITALS) + ' ,,']

def _csv_lines(header_codes, n_data, delim, field_step, field_start, field_ok, row_extra=None):
    """Lines split by '\\n' only ('\\r' and '"' outside the field grammar -> DEAD); line 0 is the literal header;
    then n_data data lines each of exactly 5 fields separated by `delim`; at most one trailing '\\n'.
    field_step/field_start/field_ok define the per-field automaton (fs = field state); row_extra(fs_acc, flag)
    lets a row-level flag be accumulated (used for the special-character requirement)."""
    H = len(header_codes)

    def step(s, c):
        line, pos, fs, flag = s
        if c == 13:
            return DEAD
        if line == 0:
            if c == 10:
                return (1, 0, field_start, flag) if pos == H else DEAD
            return (0, pos + 1, fs, flag) if pos < H and c == header_codes[pos] else DEAD
        if line > n_data:
            return DEAD
        if c == 10:
            if pos == 4 and field_ok(fs):
                flag = row_extra(fs, flag) if row_extra else flag
                return (line + 1, 0, field_start, flag)
            return DEAD
        if c == delim:
            if pos < 4 and field_ok(fs):
                flag = row_extra(fs, flag) if row_extra else flag
                return (line, pos + 1, field_start, flag)
            return DEAD
        fs2 = field_step(fs, c)
        return DEAD if fs2 is None else (line, pos, fs2, flag)

    def accept(s):
        line, pos, fs, flag = s
        if line == n_data + 1 and pos == 0 and (fs == field_start):
            return bool(flag)
        if line == n_data and pos == 4 and field_ok(fs):
            return bool(row_extra(fs, flag) if row_extra else flag)
        return False
    return build((0, 0, field_start, True if row_extra is None else False), step, accept)

@register('custom:csv_city', note='sufficient: header line literal ID,Country,City,Year,Count; 7 lines of 5 unquoted fields (no " , \\r \\n in fields); <=1 trailing newline')
def csv_city(kw, prompt):

    def fstep(fs, c):
        return None if c in (34, 44) else 'u'
    return [_csv_lines(_ascii_codes('ID,Country,City,Year,Count'), 7, 44, fstep, 'u', lambda fs: True)]

@positives('custom:csv_city')
def csv_city_pos(kw, prompt):
    rows = ['ID,Country,City,Year,Count', '1,France,Paris,2020,120', '2,Japan,Tokyo,2021,340', '3,Brazil,Sao Paulo,2019,88', '4,Canada,Toronto,2022,57', '5,Kenya,Nairobi,2018,210', '6,Italy,Rome,2023,95', '7,India,Mumbai,2020,400']
    return ['\n'.join(rows), '\n'.join(rows) + '\n', '\n'.join((r.replace(',', ', ') if i else r for i, r in enumerate(rows)))]

def _special_class(c):
    """[^\\d\\w\\s] on the class alphabet: not \\w (covers \\d), not whitespace."""
    return not is_word_class(c) and (not is_space_class(c))

@register('custom:csv_special_character', note='sufficient: header literal ProductID,Category,Brand,Price,Stock; 14 lines x 5 fields; a field is either unquoted (no " , \\r \\n) or "content" with content free of " , \\r \\n; at least one quoted field contains a non-\\w non-space char (the checker triples quotes before parsing)')
def csv_special_character(kw, prompt):

    def fstep(fs, c):
        if c == 44:
            return None
        if fs == 'e':
            if c == 34:
                return ('q', False)
            return 'u'
        if fs == 'u':
            return None if c == 34 else 'u'
        if fs[0] == 'q':
            if c == 34:
                return ('qc', fs[1])
            return ('q', fs[1] or _special_class(c))
        return None

    def fok(fs):
        return fs in ('e', 'u') or fs[0] == 'qc'

    def extra(fs, flag):
        return flag or (fs not in ('e', 'u') and fs[0] == 'qc' and fs[1])
    return [_csv_lines(_ascii_codes('ProductID,Category,Brand,Price,Stock'), 14, 44, fstep, 'e', fok, extra)]

@positives('custom:csv_special_character')
def csv_special_character_pos(kw, prompt):
    rows = ['ProductID,Category,Brand,Price,Stock'] + [f'{100 + i},Electronics,Brand{i},{19.99 + i:.2f},{10 * i}' for i in range(14)]
    a = list(rows)
    a[3] = '103,"Home & Kitchen",Brand3,22.99,30'
    b = list(rows)
    b[14] = '113,Toys,"Brand #13",32.99,130'
    d = list(rows)
    d[1] = '"100","Electronics","Brand-0","19.99","0"'
    return ['\n'.join(a), '\n'.join(b) + '\n', '\n'.join(d)]

@register('custom:csv_quotes', note='sufficient: header literal "StudentID"\\t"Subject"\\t"Grade"\\t"Semester"\\t"Score"; 3 lines of 5 fields "content" (content free of " \\t \\r \\n), single tabs; <=1 trailing newline')
def csv_quotes(kw, prompt):

    def fstep(fs, c):
        if c == 9:
            return None
        if fs == 'e':
            return 'q' if c == 34 else None
        if fs == 'q':
            return 'qc' if c == 34 else 'q'
        return None
    return [_csv_lines(_ascii_codes('"StudentID"\t"Subject"\t"Grade"\t"Semester"\t"Score"'), 3, 9, fstep, 'e', lambda fs: fs == 'qc')]

@positives('custom:csv_quotes')
def csv_quotes_pos(kw, prompt):
    rows = ['"StudentID"\t"Subject"\t"Grade"\t"Semester"\t"Score"', '"S001"\t"Mathematics"\t"A"\t"Fall 2023"\t"95"', '"S002"\t"Physics"\t"B+"\t"Spring 2024"\t"88"', '"S003"\t"History, Modern"\t"A-"\t"Fall 2024"\t"91"']
    return ['\n'.join(rows), '\n'.join(rows) + '\n']
_AFRICA = sorted(['Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 'Cape Verde', 'Central African Republic', 'Chad', 'Comoros', 'Congo', "Cote d'Ivoire", 'Democratic Republic of the Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'], reverse=True)
assert _AFRICA[0] == 'Zimbabwe' and len(_AFRICA) == 54 and all((a > b for a, b in zip(_AFRICA, _AFRICA[1:])))
_PAD = PUNCT | {32}

@register('custom:reverse_newline', note='sufficient: ASCII-only; optional preamble lines not containing "Zimbabwe"; then the fixed list of 54 African countries in reverse alphabetical order (a prefix of >=52 suffices), one per line as pad* Name pad* (pad = ASCII punctuation or space), blank/pad-only lines ignored')
def reverse_newline(kw, prompt):
    names = [tuple(map(ord, n)) for n in _AFRICA]
    ZIM = names[0]
    fz = kmp_table(ZIM)
    NEED = 52

    def line_step(ls, idx, c):
        """ls: 'lead' | ('m', k) | 'trail' for matching pad* names[idx] pad*; None = fail."""
        if ls == 'lead':
            if c in _PAD:
                return 'lead'
            return ('m', 1) if idx < len(names) and c == names[idx][0] else None
        if ls == 'trail':
            return 'trail' if c in _PAD else None
        k = ls[1]
        if k == len(names[idx]):
            return 'trail' if c in _PAD else None
        return ('m', k + 1) if c == names[idx][k] else None

    def line_done(ls, idx):
        return ls == 'trail' or (isinstance(ls, tuple) and ls[1] == len(names[idx]))

    def step(s, c):
        ph, idx, ls, zk = s
        if c >= 128 or c == 13:
            return DEAD
        if ph == 'pre':
            if c == 10:
                if line_done(ls, 0):
                    return ('list', 1, 'lead', 0)
                if zk == len(ZIM):
                    return DEAD
                return ('pre', 0, 'lead', 0)
            zk2 = kmp_advance(ZIM, fz, zk, c)
            ls2 = line_step(ls, 0, c) if ls is not None else None
            return ('pre', 0, ls2, zk2)
        if c == 10:
            if ls == 'lead':
                return ('list', idx, 'lead', 0)
            if line_done(ls, idx):
                return ('list', idx + 1, 'lead', 0)
            return DEAD
        ls2 = line_step(ls, idx, c)
        return DEAD if ls2 is None else ('list', idx, ls2, 0)

    def accept(s):
        ph, idx, ls, zk = s
        if ph != 'list':
            return False
        done = idx + (1 if ls != 'lead' and line_done(ls, idx) else 0)
        if ls != 'lead' and (not line_done(ls, idx)):
            return False
        return done >= NEED
    return [build(('pre', 0, 'lead', 0), step, accept)]

@positives('custom:reverse_newline')
def reverse_newline_pos(kw, prompt):
    return ['\n'.join(_AFRICA), '\n'.join(('- ' + n for n in _AFRICA)) + '\n', 'Here is the list:\n\n' + '\n'.join(_AFRICA[:53]), '\n'.join(('* ' + n + '.' for n in _AFRICA))]
_PALINDROMES = ['level', 'radar', 'civic', 'kayak', 'madam', 'refer', 'rotor', 'tenet', 'stats', 'solos', 'sagas', 'minim', 'sexes', 'dewed', 'redder', 'deified', 'racecar', 'reviver', 'repaper', 'rotator', 'redivider', 'detartrated', 'malayalam', 'rotavator', 'tattarrattat', 'deleveled', 'releveler', 'evitative', 'murdrum', 'terret', 'denned', 'peeped', 'pullup', 'succus', 'hallah', 'reifier', 'aibohphobia', 'shahs', 'siris', 'kaiak', 'lemel', 'sedes', 'serer', 'noon', 'deed', 'toot', 'peep', 'sees', 'anna', 'otto', 'abba', 'wow', 'eye', 'did', 'pop', 'mom', 'dad']
_PALINDROMES = [w for w in _PALINDROMES if w == w[::-1] and len(w) >= 5]

def _trie(words):
    """Returns (children: dict[node][code] -> node, terminal: dict[node] -> word_index)."""
    children, terminal = ({0: {}}, {})
    for wi, w in enumerate(words):
        node = 0
        for ch in w:
            code = ord(ch)
            if code not in children[node]:
                nxt = len(children)
                children[node][code] = nxt
                children[nxt] = {}
            node = children[node][code]
        terminal[node] = wi
    return (children, terminal)

@register('words:palindrome', note='sufficient: counts whitespace-separated tokens that, after dropping ASCII punctuation and lower(), spell one of %d fixed palindromes (>=5 chars); other palindromes are not counted' % len(_PALINDROMES))
def palindrome(kw, prompt):
    children, terminal = _trie(_PALINDROMES)
    NEED = 10

    def step(s, c):
        node, cnt = s
        if is_space_class(c):
            if node > 0 and node in terminal:
                cnt = min(NEED, cnt + 1)
            return (0, cnt)
        if c in PUNCT:
            return s
        if node < 0:
            return s
        if c >= 128:
            return (-1, cnt)
        nxt = children[node].get(fold_lower(c))
        return (nxt if nxt is not None else -1, cnt)

    def accept(s):
        node, cnt = s
        return cnt + (1 if node > 0 and node in terminal else 0) >= NEED
    return [build((0, 0), step, accept)]

@positives('words:palindrome')
def palindrome_pos(kw, prompt):
    return ['The radar on the kayak showed a level civic rotor; madam, refer to the tenet of stats and solos.', 'Level, radar, civic, kayak, madam, refer, rotor, tenet, stats, solos, sagas, minim.', 'A racecar driver named Madam Hannah (who deified her rotor) said: "Refer to the level radar, the civic kayak, the tenet, the stats!"\n\nSolos and sagas.', 'LEVEL RADAR CIVIC KAYAK MADAM REFER ROTOR TENET STATS SOLOS SAGAS MINIM SEXES DEWED']
_NAMES = ['Emma', 'Liam', 'Sophia', 'Jackson', 'Olivia', 'Noah', 'Ava', 'Lucas', 'Isabella', 'Mason', 'Mia', 'Ethan', 'Charlotte', 'Alexander', 'Amelia', 'Benjamin', 'Harper', 'Leo', 'Zoe', 'Daniel', 'Chloe', 'Samuel', 'Lily', 'Matthew', 'Grace', 'Owen', 'Abigail', 'Gabriel', 'Ella', 'Jacob', 'Scarlett', 'Nathan', 'Victoria', 'Elijah', 'Layla', 'Nicholas', 'Audrey', 'David', 'Hannah', 'Christopher', 'Penelope', 'Thomas', 'Nora', 'Andrew', 'Aria', 'Joseph', 'Claire', 'Ryan', 'Stella', 'Jonathan']

def _person_names_dfa(N, G, S):
    """Names counted greedily in increasing list order: from pointer p, a word-bounded (\\b, Unicode \\w) exact
    occurrence of names[j] with p <= j <= p+G advances p = j+1 and the count; at most S names skipped in total."""
    n = len(_NAMES)
    tries = {}
    for p in range(n):
        tries[p] = _trie(_NAMES[p:min(n, p + G + 1)])

    def step(s, c):
        p, cnt, node, prevw = s
        wc = is_word_class(c)
        if cnt >= N:
            return (p, cnt, 0, wc)
        children, terminal = tries[p] if p < n else ({0: {}}, {})
        if node > 0 and node in terminal and (not wc):
            j = p + terminal[node]
            p2 = j + 1
            if p2 - (cnt + 1) > S:
                return DEAD
            return (p2, cnt + 1, 0, False)
        if not wc:
            return (p, cnt, 0, False)
        if node > 0:
            nxt = children[node].get(c)
            return (p, cnt, nxt if nxt is not None else -1, True)
        if node == 0 and (not prevw) and (p < n):
            nxt = children[0].get(c)
            return (p, cnt, nxt if nxt is not None else -1, True)
        return (p, cnt, -1 if node == 0 else node, True)

    def accept(s):
        p, cnt, node, prevw = s
        if cnt >= N:
            return True
        if node > 0 and p < n and (node in tries[p][1]):
            j = p + tries[p][1][node]
            return cnt + 1 >= N and j + 1 - (cnt + 1) <= S
        return False
    return build((0, 0, 0, False), step, accept)

@register('count:person_names', note='sufficient: distinct names counted only along increasing list order (gap <= G between consecutive counted names, <= S skipped in total; G,S tuned per N to keep the DFA under ~6k states); \\b-bounded case-sensitive matches as in the checker')
def person_names(kw, prompt):
    N = int(kw['N'])
    if N <= 0:
        return [build(0, lambda s, c: 0, lambda s: True)]
    if N > len(_NAMES):
        raise Unsupported('N exceeds the number of names')
    best = None
    for G, S in ((6, 50), (5, 50), (4, 50), (6, 14), (4, 14), (3, 14), (3, 10), (2, 10), (2, 6), (1, 6), (1, 3), (0, 0)):
        d = _person_names_dfa(N, G, min(S, len(_NAMES) - N))
        if len(d.states) <= 6000:
            best = d
            break
    return [best]

@positives('count:person_names')
def person_names_pos(kw, prompt):
    N = int(kw['N'])
    first = _NAMES[:N]
    s1 = 'At the reunion, ' + ', '.join(first[:-1]) + ' and ' + first[-1] + ' shared stories about the summer.'
    s2 = '\n'.join((f'{i + 1}. {nm} brought a dish.' for i, nm in enumerate(first)))
    s3 = ' '.join((f"{nm}'s laughter filled the room;" for nm in _NAMES[1:1 + N])) + ' everyone agreed.'
    skipped = [_NAMES[i] for i in range(0, len(_NAMES), 2)][:N]
    s4 = 'Guests: ' + ', '.join(skipped) + '.' if len(skipped) >= N else s1
    return [s1, s2, s3, s4]
_ANCHORS = ['the', 'i', 'we', 'it', 'this', 'life', 'love', 'time', 'today', 'hope', 'dreams', 'home', 'music', 'nature', 'change', 'water', 'light', 'people', 'work', 'now', 'again', 'you', 'always', 'together', 'freedom', 'peace', 'code', 'data', 'python', 'learning', 'knowledge', 'science', 'history', 'books', 'words', 'stories', 'art', 'food', 'family', 'friends', 'money', 'health', 'happiness', 'success', 'future', 'world', 'earth', 'sun', 'moon', 'stars', 'ocean', 'city', 'rain', 'fire', 'journey', 'silence', 'truth', 'beauty', 'power', 'energy', 'technology', 'travel', 'coffee', 'night', 'day', 'morning', 'children', 'school', 'reading', 'writing', 'summer', 'winter']

@register('words:paragraph_last_first', note="sufficient: lines with <=1 token always pass; a line with >=2 tokens must start with one of %d anchor words (after the checker's strip/lower/punct-strip) and end with the same word; the trailing junk after it must be ws* (punct|space)* ws* so the checker strips it" % len(_ANCHORS))
def paragraph_last_first(kw, prompt):
    children, terminal = _trie(_ANCHORS)
    W = [tuple(map(ord, w)) for w in _ANCHORS]

    def tstep(node, c):
        if node < 0:
            return -1
        nxt = children[node].get(lower_strict(c))
        return -1 if nxt is None else nxt

    def extend(m, wid, c):
        if m == 'FAIL' or wid < 0:
            return 'FAIL'
        return m + 1 if m < len(W[wid]) and lower_strict(c) == W[wid][m] else 'FAIL'
    PAD_T = {'INRUN': 'S1F', 'S0': 'S1Tp', 'S1F': 'S1F', 'S1Tp': 'S1Tp', 'S1Ts': 'S1Tp', 'S2': 'BRp', 'BRp': 'BRp', 'BRw': 'BRp'}
    SP_T = {'INRUN': 'S0', 'S0': 'S0', 'S1F': 'S1Ts', 'S1Tp': 'S1Ts', 'S1Ts': 'S1Ts', 'S2': 'S2', 'BRp': 'BRw', 'BRw': 'BRw'}
    WS_T = {'INRUN': 'S0', 'S0': 'S0', 'S1F': 'S2', 'S1Tp': 'S2', 'S1Ts': 'S2', 'S2': 'S2', 'BRp': 'BRw', 'BRw': 'BRw'}

    def line_pass(st):
        if st == 'LW0':
            return False
        if st[0] != 'B':
            return True
        _, wid, m, tail = st
        full = m == 'FIRST' or (wid >= 0 and m == len(W[wid]))
        return full and tail not in ('BRp', 'BRw')

    def step(st, c):
        if c == 10:
            if st == 'LW0':
                return 'LW0'
            return 'LW' if line_pass(st) else DEAD
        ws, sp, pad = (is_space_class(c), c == 32, c in PUNCT)
        tok = not ws and (not pad)
        if st in ('LW', 'LW0'):
            if ws:
                return st
            if pad:
                return 'LP'
            return ('R1', tstep(0, c))
        if st == 'LP':
            if pad or sp:
                return 'LP'
            if ws:
                return 'PRE'
            return ('R1', tstep(0, c))
        if st == 'PRE':
            if ws:
                return 'PRE'
            if pad:
                return ('R1', -1)
            return ('R1', tstep(0, c))
        if st[0] == 'R1':
            node = st[1]
            if ws:
                return ('B', terminal.get(node, -1) if node > 0 else -1, 'FIRST', 'S0')
            if pad:
                return ('R1', -1)
            return ('R1', tstep(node, c))
        _, wid, m, tail = st
        if tok:
            if tail == 'INRUN':
                m2 = extend(m, wid, c)
            elif tail in ('S1F', 'S1Tp', 'BRp'):
                m2 = 'FAIL'
            else:
                if wid < 0:
                    return DEAD
                m2 = extend(0, wid, c)
            return ('B', wid, m2, 'INRUN')
        t2 = (PAD_T if pad else SP_T if sp else WS_T)[tail]
        return ('B', wid, m, t2)
    return [build('LW0', step, line_pass)]

@positives('words:paragraph_last_first')
def paragraph_last_first_pos(kw, prompt):
    return ['Time flies when you are having fun, and nobody can stop time.\nLife is what happens while you plan your life.\n\nHope is the thing with feathers, said the poet of hope.', 'The quick brown fox jumps over the lazy dog, and so does the.\n\nMusic soothes the soul; nothing beats music!', 'Water.\nPeople need water, and water needs people.\n   Love conquers all, so give love...  \nFreedom', 'I think, therefore I.\nWe are what we repeatedly do, said we.\nIt is what it is: it.']
_TB_SPLIT_WORDS = ['cannot', 'gimme', 'gonna', 'gotta', 'lemme', 'wanna']
_OPEN, _CLOSE, _TRAIL = (frozenset(map(ord, '("')), frozenset(map(ord, ')"')), frozenset(map(ord, ',;:!?')))

def _is_ascii_letter(c):
    return 65 <= c <= 90 or 97 <= c <= 122

@register('words:words_position', note='sufficient: ASCII-only; whitespace-separated runs OPEN? WORD CLOSE? TRAIL? (OPEN ( ", CLOSE ) ", TRAIL , ; : ! ? ., also TRAIL CLOSE) with WORD = letters+ | digits+, or punctuation runs - -- * ( ) "; each run is exactly one alnum token for nltk.word_tokenize (verified for all combos); cannot/gimme/gonna/gotta/lemme/wanna are rejected (Treebank splits them); word #2 and the second-to-last word must be the keyword without a trailing period')
def words_position(kw, prompt):
    key = kw['keyword'].strip().lower()
    if not key or not (all(('a' <= ch <= 'z' for ch in key)) or all(('0' <= ch <= '9' for ch in key))):
        raise Unsupported('keyword must be ASCII letters-only or digits-only for the word_tokenize sublanguage')
    K = tuple(map(ord, key))
    L = len(K)
    fch, fterm = _trie(_TB_SPLIT_WORDS)

    def complete(s, iskw, fnode):
        n, w1, prev, last = s
        if fnode > 0 and fnode in fterm:
            return None
        n2 = min(2, n + 1)
        return (n2, iskw if n2 == 2 and n == 1 else w1, last, iskw)

    def kw_adv(k, c):
        lc = lower_strict(c)
        return k + 1 if k >= 0 and k < L and (lc == K[k]) else -1

    def f_adv(f, c):
        if f < 0:
            return -1
        nxt = fch[f].get(lower_strict(c))
        return -1 if nxt is None else nxt

    def step(st, c):
        s, run = st
        if c >= 128:
            return DEAD
        ws = is_space_class(c)
        letter, digit = (_is_ascii_letter(c), 48 <= c <= 57)
        if ws:
            if run in ('gap', 'dash1', 'punct', 'open'):
                return (s, 'gap')
            kind = run[0]
            if kind == 'L':
                s2 = complete(s, run[1] == L, run[2])
            elif kind == 'D':
                s2 = complete(s, False, -1)
            else:
                s2 = complete(s, run[1], -1)
            return DEAD if s2 is None else (s2, 'gap')
        if run == 'gap':
            if c in _OPEN:
                return (s, 'open')
            if letter:
                return (s, ('L', kw_adv(0, c), f_adv(0, c)))
            if digit:
                return (s, ('D',))
            if c == 45:
                return (s, 'dash1')
            if c == 42 or c == 41:
                return (s, 'punct')
            return DEAD
        if run == 'open':
            if letter:
                return (s, ('L', kw_adv(0, c), f_adv(0, c)))
            if digit:
                return (s, ('D',))
            return DEAD
        if run == 'dash1':
            return (s, 'punct') if c == 45 else DEAD
        if run == 'punct':
            return DEAD
        kind = run[0]
        if kind == 'L':
            if letter:
                return (s, ('L', kw_adv(run[1], c), f_adv(run[2], c)))
            if run[2] > 0 and run[2] in fterm:
                return DEAD
            iskw = run[1] == L
            if c in _CLOSE:
                return (s, ('C', iskw))
            if c in _TRAIL:
                return (s, ('T', iskw))
            if c == 46:
                return (s, ('T', False))
            return DEAD
        if kind == 'D':
            if digit:
                return st
            if c in _CLOSE:
                return (s, ('C', False))
            if c in _TRAIL or c == 46:
                return (s, ('T', False))
            return DEAD
        if kind == 'C':
            if c in _TRAIL:
                return (s, ('T', run[1]))
            if c == 46:
                return (s, ('T', False))
            return DEAD
        if kind == 'T' and c in _CLOSE:
            return (s, ('TC', run[1]))
        return DEAD

    def accept(st):
        s, run = st
        if run not in ('gap', 'dash1', 'punct', 'open'):
            kind = run[0]
            if kind == 'L':
                s = complete(s, run[1] == L, run[2])
            elif kind == 'D':
                s = complete(s, False, -1)
            else:
                s = complete(s, run[1], -1)
            if s is None:
                return False
        n, w1, prev, last = s
        return n >= 2 and w1 and prev
    return [build(((0, False, False, False), 'gap'), step, accept)]

@positives('words:words_position')
def words_position_pos(kw, prompt):
    k = kw['keyword'].strip()
    return [f'A {k} garden grows behind the old house, and every morning the colors look {k} again.', f'Stay {k}, my friend: the city never sleeps, so remain {k} always!', f'The {k} idea came first.\n\n- Point one\n- Point two\n\nIt stayed {k} forever.', f'Truly {k} {k}.', f'"Be {k}," she said (quietly), "and keep it {k} today."']
_PRONOUNS = ['i', 'me', 'we', 'us', 'you', 'he', 'him', 'she', 'her', 'it', 'they', 'them', 'my', 'mine', 'our', 'ours', 'your', 'yours', 'his', 'hers', 'its', 'their', 'theirs', 'myself', 'ourselves', 'yourself', 'yourselves', 'himself', 'herself', 'itself', 'themselves', 'this', 'that', 'these', 'those', 'who', 'whom', 'whose', 'which', 'what', 'whoever', 'whomever', 'whatever', 'whichever', 'anybody', 'anyone', 'anything', 'everybody', 'everyone', 'everything', 'nobody', 'nothing', 'somebody', 'someone', 'something', 'each', 'either', 'neither', 'both', 'all', 'some', 'any', 'none']
_P1 = frozenset(map(ord, ',;:!?'))

@register('count:pronouns', note='sufficient: a pronoun counts only when it is an ASCII-letter run preceded by start/ASCII whitespace/"/" (or ( " after such a delimiter) and followed by whitespace, "/", end, ) ", a final period (then only whitespace to the end), one of , ; : ! ? then whitespace/end/"/", or a Treebank contraction suffix \'s \'m \'d \'re \'ll \'ve; all contexts verified against nltk.word_tokenize (3.10); pronouns next to periods, dashes or non-ASCII are not counted')
def pronouns(kw, prompt):
    N = int(kw['N'])
    if N <= 0:
        return [build(0, lambda s, c: 0, lambda s: True)]
    children, terminal = _trie(_PRONOUNS)
    AP = {0: {115: 'done', 109: 'done', 100: 'done', 114: 'r', 108: 'l', 118: 'v'}, 'r': {101: 'done'}, 'l': {108: 'done'}, 'v': {101: 'done'}}

    def delim_ws(c):
        return c < 128 and is_space_class(c) or c == 47

    def step(s, c):
        cnt, ph = s
        if cnt >= N:
            return (N, 'sink')
        if c >= 128:
            return (cnt, 'gap_no')
        if ph in ('gap_ok', 'gap_no'):
            if delim_ws(c):
                return (cnt, 'gap_ok')
            if ph == 'gap_ok':
                if c in (40, 34):
                    return (cnt, 'gap_ok')
                if _is_ascii_letter(c):
                    nxt = children[0].get(lower_strict(c))
                    return (cnt, ('T', nxt)) if nxt is not None else (cnt, 'gap_no')
            return (cnt, 'gap_no')
        if ph == 'P1':
            if delim_ws(c):
                return (cnt + 1, 'gap_ok')
            if c == 34:
                return (cnt + 1, 'gap_no')
            return (cnt, 'gap_no')
        if ph == 'APdone':
            if delim_ws(c):
                return (cnt + 1, 'gap_ok')
            if c in _P1:
                return (cnt, 'P1')
            if c == 41:
                return (cnt + 1, 'gap_no')
            if c == 46:
                return (cnt, 'PEND')
            return (cnt, 'gap_no')
        if ph == 'PEND':
            return (cnt, 'PENDws') if delim_ws(c) else (cnt, 'gap_no')
        if ph == 'PENDws':
            if delim_ws(c):
                return (cnt, 'PENDws')
            if c in (40, 34):
                return (cnt, 'gap_ok')
            if _is_ascii_letter(c):
                nxt = children[0].get(lower_strict(c))
                return (cnt, ('T', nxt)) if nxt is not None else (cnt, 'gap_no')
            return (cnt, 'gap_no')
        if ph[0] == 'AP':
            nxt = AP[ph[1]].get(lower_strict(c))
            if nxt is None:
                return (cnt, 'gap_no')
            return (cnt, 'APdone') if nxt == 'done' else (cnt, ('AP', nxt))
        node = ph[1]
        if _is_ascii_letter(c):
            nxt = children[node].get(lower_strict(c))
            return (cnt, ('T', nxt)) if nxt is not None else (cnt, 'gap_no')
        if node in terminal:
            if delim_ws(c):
                return (cnt + 1, 'gap_ok')
            if c == 34:
                return (cnt + 1, 'gap_ok')
            if c == 41:
                return (cnt + 1, 'gap_no')
            if c in _P1:
                return (cnt, 'P1')
            if c == 39:
                return (cnt, ('AP', 0))
            if c == 46:
                return (cnt, 'PEND')
        return (cnt, 'gap_no')

    def accept(s):
        cnt, ph = s
        pend = ph in ('P1', 'APdone', 'PEND', 'PENDws') or (isinstance(ph, tuple) and ph[0] == 'T' and (ph[1] in terminal))
        return cnt + (1 if pend else 0) >= N
    return [build((0, 'gap_ok'), step, accept)]

@positives('count:pronouns')
def pronouns_pos(kw, prompt):
    N = int(kw['N'])
    base = "I told them that we could do it ourselves, but she said her plan was better than mine. They gave us everything; nobody knew who did what, which is why everyone blamed himself. It's yours now: this, that, these and those are all his, hers or theirs! "
    reps = max(1, (N + 25) // 26)
    return [base * reps, (base.lower() * reps).strip(), 'She/her/hers. ' + base * reps, '\n'.join((f'- {w.capitalize()} matters.' for w in _PRONOUNS[:max(N, 1)])), ' '.join(_PRONOUNS[:N]) + '.']

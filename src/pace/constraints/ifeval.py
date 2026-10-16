import string
import numpy as np
from pace.automata.alphabet import (
    N_CLASSES,
    is_space_class,
    is_word_class,
    is_decimal_class,
    fold_rei,
)
import pace.constraints.base as PV
from pace.constraints.base import (
    DFA,
    DEAD,
    Unsupported,
    build,
    nonempty,
    product,
    minimize,
    kmp_table,
    kmp_advance,
    lower_strict,
    literal_count,
    letter_count,
    case_all,
    word_count,
    no_comma,
    quotation,
    ends_with,
    constrained_response,
    bullets,
    paragraphs,
    sections,
    nth_paragraph_first_word,
)

BLOCK_MIN_WORDS = 8
PARAGRAPH_MIN_WORDS = 12
BULLET_MIN_WORDS = 4
REPEAT_MIN_WORDS = 20
POSTSCRIPT_MIN_WORDS = 4
CASE_MIN_WORDS = 10
CHAR_RUN_LIMIT = 6
STAR_RUN_LIMIT = 7
FORBIDDEN_GLUE_MODE = "affix"
FORBIDDEN_GLUE_MIN_LEN = 4
LETTER_RUN_CAP = 6
TRIPLE_EXEMPT = frozenset(map(ord, "wi"))
FILLER_TITLES = (
    "title",
    "your title",
    "insert title",
    "title here",
    "your title here",
    "insert title here",
    "poem title",
    "the title",
    "title of the poem",
    "title goes here",
    "enter title",
    "untitled",
)
HIGHLIGHT_FORBIDDEN_LITERAL = "highlighted section"
NONASCII_LETTER = frozenset((128, 129, 130, 131, 133, 134, 135, 140))
_RUN_EXEMPT = frozenset(
    (c for c in range(128) if chr(c).isspace() or chr(c).isdigit())
) | frozenset(map(ord, "-=_"))
CHAR_RUN_TRACKED = frozenset([42])


def is_ascii_letter(c):
    return 65 <= c <= 90 or 97 <= c <= 122


def is_letter_class(c):
    return is_ascii_letter(c) or c in NONASCII_LETTER


def fold_ascii(c):
    """ASCII-only case fold: non-ASCII classes never fold onto ASCII letters."""
    return c + 32 if 65 <= c <= 90 else c


def _ascii_word(word):
    return bool(word) and word.isascii() and word.isalpha()


def no_homoglyph():
    """No \\w-run contains both an ASCII letter and a non-ASCII letter class."""

    def step(s, c):
        if not is_word_class(c):
            return (False, False)
        a, n = s
        a = a or is_ascii_letter(c)
        n = n or c in NONASCII_LETTER
        return DEAD if a and n else (a, n)

    return build((False, False), step, lambda s: True)


def no_char_runs(limit=CHAR_RUN_LIMIT, star_limit=STAR_RUN_LIMIT, tracked=None):
    """No tracked ASCII character repeated `limit` times in a row ('*': star_limit). `tracked` = None tracks every
    ASCII character except whitespace, digits and - = _ (528 states: the state remembers the last character, which
    multiplies with every other component in the joint product); CHAR_RUN_TRACKED (default) tracks only the junk-run
    characters seen in the audits. Non-ASCII classes are not single characters, so they are never tracked."""
    tracked = CHAR_RUN_TRACKED if tracked is None else tracked

    def step(s, c):
        if c >= 128 or c in _RUN_EXEMPT or (tracked != "all" and c not in tracked):
            return (None, 0)
        last, run = s
        run = run + 1 if last == c else 1
        if run >= (star_limit if c == 42 else limit):
            return DEAD
        return (c, run)

    return build((None, 0), step, lambda s: True)


def no_literal(text):
    """The literal (re.I, substring) must not occur anywhere."""
    return literal_count(text, 1, "less than", False)


def no_classes(classes):
    classes = frozenset(classes)
    return build(0, lambda s, c: DEAD if c in classes else 0, lambda s: True)


def keyword_whole_ascii(word, threshold):
    """Count >= threshold whole-word occurrences of `word` whose characters are ASCII letters (case-insensitive
    ASCII fold). Subset of the official substring/re.I count, so sound for 'at least'."""
    if not _ascii_word(word):
        raise Unsupported("keyword must be ASCII letters")
    W = tuple((ord(ch) for ch in word.lower()))
    L, cap = (len(W), max(threshold, 1))

    def step(s, c):
        prefix, count, prev_word = s
        fc, wc = (fold_ascii(c), is_word_class(c))
        if prefix == L and (not wc):
            count = min(cap, count + 1)
        if 0 < prefix < L and fc == W[prefix]:
            ap = prefix + 1
        elif not prev_word and fc == W[0]:
            ap = 1
        else:
            ap = 0
        return (ap, count, wc)

    def accept(s):
        prefix, count, _ = s
        return min(cap, count + int(prefix == L)) >= threshold

    return build((0, 0, False), step, accept)


def forbidden_glue(word, mode=FORBIDDEN_GLUE_MODE):
    """Forbid `word` (re.I fold) glued to a longer token: as a prefix (\\bWORD\\w) or suffix (\\wWORD\\b) in
    'affix' mode, or every occurrence in 'substring' mode. Combined with the official \\bWORD\\b DFA."""
    if mode == "substring":
        return literal_count(word, 1, "less than", False)
    if not _ascii_word(word):
        raise Unsupported("forbidden glue needs an ASCII-letter word")
    pat = tuple((ord(ch) for ch in word.lower()))
    f, L = (kmp_table(pat), len(pat))

    def step(s, c):
        k, left_glued, prev_word = s
        wc = is_word_class(c)
        if k == L and (not wc):
            return DEAD
        k2 = kmp_advance(pat, f, k, fold_rei(c))
        if k2 == 0:
            lg = False
        elif k2 == k + 1 and k > 0:
            lg = left_glued
        elif k2 == 1 and k2 == k + 1:
            lg = prev_word
        else:
            lg = True
        if k2 == L and (not lg):
            return DEAD
        return (k2, lg, wc)

    return build((0, False, False), step, lambda s: s[0] < L)


def triple_letter_forbid(letters=None, exempt=TRIPLE_EXEMPT):
    """No ASCII letter (case-insensitive) in `letters` repeated 3 times consecutively ('llll'). letters=None tracks
    every letter except the exempt ones (50 states); the default call tracks only the instruction's target letter
    (3 states), which is what the padding loophole adds."""
    tracked = None if letters is None else frozenset((ord(ch) for ch in letters.lower()))

    def step(s, c):
        if not is_ascii_letter(c):
            return (None, 0)
        fc = fold_ascii(c)
        if tracked is None and fc in exempt or (tracked is not None and fc not in tracked):
            return (None, 0)
        last, run = s
        run = run + 1 if last == fc else 1
        return DEAD if run >= 3 else (fc, run)

    return build((None, 0), step, lambda s: True)


def letter_count_alpha(letter, threshold):
    """'at least' letter count restricted to ASCII letters in the alphabetic prefix of a word run: once a run contains a
    digit, '_' or a non-ASCII word character, its remaining letters do not count ('x1x1x1' counts one x). A sound
    undercount of the official Counter(value.lower()) in 3(N+1) states (the exact 'alphabetic words only' version
    needs a per-run pending counter and multiplied the joint product by ~10)."""
    if len(letter) != 1 or not "a" <= letter.lower() <= "z":
        raise Unsupported("letter_frequency with non-letter uses a random official fallback")
    target = ord(letter.lower())
    cap = max(threshold, 1)

    def step(s, c):
        total, run = s
        if is_ascii_letter(c):
            if run == 2:
                return s
            return (min(cap, total + int(fold_ascii(c) == target)), 1)
        if is_word_class(c):
            return (total, 2)
        return (total, 0)

    return build((0, 0), step, lambda s: s[0] >= threshold)


def placeholders_hard(n):
    """Mirror of re.findall(r'\\[.*?\\]') counting only placeholders with >= 2 word chars; '[' inside, '"' inside,
    '[]', and directly adjacent '][' are dead."""

    def step(s, c):
        count, inside, letters, just_closed = s
        if inside:
            if c == 93:
                if letters < 2:
                    return DEAD
                return (min(n, count + 1), False, 0, True)
            if c == 10:
                return (count, False, 0, False)
            if c == 91 or c == 34:
                return DEAD
            return (count, True, min(2, letters + int(is_word_class(c))), False)
        if c == 91:
            if just_closed:
                return DEAD
            return (count, True, 0, False)
        return (count, False, 0, False)

    return build((0, False, 0, False), step, lambda s: s[0] >= n)


def _trie(phrases):
    children, terminal = ([{}], set())
    for p in phrases:
        node = 0
        for ch in p:
            nxt = children[node].get(ord(ch))
            if nxt is None:
                nxt = len(children)
                children[node][ord(ch)] = nxt
                children.append({})
            node = nxt
        terminal.add(node)
    return (children, terminal)


def title_hard(fillers=FILLER_TITLES):
    """<<content>> on one line with >= 2 word runs (class 133 chars count one each) and content not a filler literal.
    Scan mirrors the official regex: the first '<<' of the line opens, '>>' closes; '<' / '>' inside are separators."""
    children, terminal = _trie(fillers)

    def tstep(t, c):
        if t == "no":
            return "no"
        if t == "lead":
            if c == 32:
                return "lead"
            nxt = children[0].get(fold_ascii(c))
            return "no" if nxt is None else nxt
        if t == "trail":
            return "trail" if c == 32 else "no"
        nxt = children[t].get(fold_ascii(c))
        if nxt is not None:
            return nxt
        if c == 32 and t in terminal:
            return "trail"
        return "no"

    def is_filler(t):
        return t == "trail" or (isinstance(t, int) and t in terminal)

    def step(s, c):
        if s == "ACC":
            return "ACC"
        if c == 10:
            return 0
        if s == 0:
            return 1 if c == 60 else 0
        if s == 1:
            return (0, False, "lead", False) if c == 60 else 0
        runs, in_run, t, pend = s
        if pend:
            if c == 62:
                return "ACC" if runs >= 2 and (not is_filler(t)) else 0
            in_run = False
            t = tstep(t, 62)
        if c == 62:
            return (runs, False, t, True)
        t = tstep(t, c)
        if c == 133:
            return (min(2, runs + 1), False, t, False)
        if is_word_class(c):
            if not in_run:
                runs = min(2, runs + 1)
            return (runs, True, t, False)
        return (runs, False, t, False)

    return build(0, step, lambda s: s == "ACC")


def highlights_hard(n):
    """Mirror of the official single/double-star scans; a span counts only if it has >= 2 word runs and (single) its
    opening '*' is not a line-start bullet marker; a span with content but no word character is dead."""
    W_CAP = 2

    def wtrack(words, in_word, c):
        wc = is_word_class(c)
        if wc and (not in_word):
            words = min(W_CAP, words + 1)
        return (words, wc)

    def step(s, c):
        total, sg, dd, ls = s
        nl, star, ws = (c == 10, c == 42, is_space_class(c))
        if sg is None:
            sg2 = (False, 0, False, "cand" if ls else "no") if star else None
        elif star:
            has_nonws, words, in_word, bullet = sg
            if has_nonws:
                if words == 0:
                    return DEAD
                if words >= 2 and bullet != "yes":
                    total = min(n, total + 1)
            sg2 = None
        elif nl:
            sg2 = None
        else:
            has_nonws, words, in_word, bullet = sg
            if bullet == "cand":
                bullet = "yes" if ws else "no"
            words, in_word = wtrack(words, in_word, c)
            sg2 = (has_nonws or not ws, words, in_word, bullet)
        if dd == 0:
            dd2 = 1 if star else 0
        elif dd == 1:
            dd2 = (2, True, False, 0, False) if star else 0
        else:
            phase, empty, good, words, in_word = dd
            if phase == 2:
                if star:
                    dd2 = (3, empty, good, words, in_word)
                elif nl:
                    dd2 = 0
                else:
                    words, in_word = wtrack(words, in_word, c)
                    dd2 = (2, False, good or not ws, words, in_word)
            elif star:
                if good:
                    if words == 0:
                        return DEAD
                    if words >= 2:
                        total = min(n, total + 1)
                dd2 = 0
            elif nl:
                dd2 = 0
            elif empty:
                words, in_word = wtrack(0, False, c)
                dd2 = (2, False, not ws, words, in_word)
            else:
                dd2 = 0
        ls2 = True if nl else ls and ws
        return (total, sg2, dd2, ls2)

    return build((0, None, 0, True), step, lambda s: s[0] >= n)


def paragraph_blocks(min_words=PARAGRAPH_MIN_WORDS):
    """Blocks split by '***' (mirrors re.split(r'\\s?\\*\\*\\*\\s?')): every block has >= min_words word runs, except an
    empty first/last block. A block with content and fewer words is dead; so is a 4th+ consecutive '*' ('****')."""
    cap = min_words

    def step(s, c):
        stars, words, in_word, content, first = s
        if c == 42:
            if stars == "sep":
                return DEAD
            if stars == 2:
                if words < cap and (content or not first):
                    return DEAD
                return ("sep", 0, False, False, False)
            return (stars + 1, words, in_word, content, first)
        if stars == "sep":
            stars = 0
        if stars:
            content = True
        wc = is_word_class(c)
        if wc and (not in_word):
            words = min(cap, words + 1)
        content = content or not is_space_class(c)
        return (0, words, wc, content, first)

    def accept(s):
        stars, words, in_word, content, first = s
        content = content or (stars != "sep" and stars > 0)
        return words >= cap or not content

    return build((0, 0, False, False, True), step, accept)


def no_star_only_lines():
    """No line consisting only of '*' and blanks other than exactly '***' (kills '**', '* **', '** * **' pseudo-dividers).
    State: 'LS' (line start) / stars seen so far on a blank-star-only line / 'TXT' (line has other content)."""

    def step(s, c):
        if c == 10:
            if s == "TXT" or s == "LS":
                return "LS"
            stars, blank = s
            return "LS" if stars == 3 else DEAD
        if s == "TXT":
            return "TXT"
        if is_space_class(c):
            if s == "LS":
                return "LS"
            stars, blank = s
            return (stars, True)
        if c == 42:
            if s == "LS":
                return (1, False)
            stars, blank = s
            if blank:
                return DEAD
            return (min(4, stars + 1), False)
        return "TXT"

    def accept(s):
        return s in ("LS", "TXT") or s[0] == 3

    return build("LS", step, accept)


def two_response_blocks(min_words=PARAGRAPH_MIN_WORDS):
    """HARDENING-ONLY: blocks split by '******' (str.split semantics) each hold >= min_words word runs; an empty
    first/last block is tolerated (official), any other short block is dead."""
    cap = min_words

    def step(s, c):
        stars, words, in_word, content, first = s
        if c == 42:
            if stars == 5:
                if words < cap and (content or not first):
                    return DEAD
                return (0, 0, False, False, False)
            return (stars + 1, words, in_word, content, first)
        if stars:
            content = True
        wc = is_word_class(c)
        if wc and (not in_word):
            words = min(cap, words + 1)
        content = content or not is_space_class(c)
        return (0, words, wc, content, first)

    def accept(s):
        stars, words, in_word, content, first = s
        content = content or stars > 0
        return words >= cap or not content

    return build((0, 0, False, False, True), step, accept)


def newline_blocks(min_words=BLOCK_MIN_WORDS):
    """Blocks split by '\\n\\n' (nth_paragraph_first_word): every block with content has >= min_words word runs."""
    cap = min_words

    def step(s, c):
        pend, words, in_word, content = s
        if c == 10:
            if pend:
                if content and words < cap:
                    return DEAD
                return (False, 0, False, False)
            return (True, words, False, content)
        wc = is_word_class(c)
        if wc and (not in_word):
            words = min(cap, words + 1)
        return (False, words, wc, content or not is_space_class(c))

    return build((False, 0, False, False), step, lambda s: s[1] >= cap or not s[3])


def bullet_lines(min_words=BULLET_MIN_WORDS):
    """Every bullet line (^\\s*\\*[^*].*$ or ^\\s*-.*$, MULTILINE) holds >= min_words word runs after the marker."""
    cap = min_words

    def step(s, c):
        ws, nl = (is_space_class(c), c == 10)
        if s in ("LS",):
            if ws:
                return "LS"
            if c == 42:
                return "STAR"
            if c == 45:
                return ("BUL", 0, False)
            return "OTHER"
        if s == "STAR":
            if c == 42:
                return "OTHER"
            if nl:
                return DEAD
            wc = is_word_class(c)
            return ("BUL", int(wc), wc)
        if s == "OTHER":
            return "LS" if nl else "OTHER"
        _, words, in_word = s
        if nl:
            return DEAD if words < cap else "LS"
        wc = is_word_class(c)
        if wc and (not in_word):
            words = min(cap, words + 1)
        return ("BUL", words, wc)

    return build("LS", step, lambda s: not (isinstance(s, tuple) and s[1] < cap))


def section_blocks(splitter, min_words=BLOCK_MIN_WORDS):
    """Blocks after each splitter match (mirrors re.split(r'\\s?SPLIT\\s?\\d+\\s?')): every section holds
    >= min_words word runs. The splitter word itself is not counted when it starts a word run."""
    pat = tuple((ord(ch) for ch in splitter))
    if (
        not pat
        or any((x > 127 for x in pat))
        or any((is_decimal_class(x) or is_space_class(x) for x in pat))
    ):
        raise Unsupported("section splitter")
    f, L = (kmp_table(pat), len(pat))
    cap = min_words + 1

    def step(s, c):
        k, ph, started, count, in_word, at_run, flag = s
        wc = is_word_class(c)
        new_run = wc and (not in_word)
        k2 = kmp_advance(pat, f, k, c)
        dig, ws = (is_decimal_class(c), is_space_class(c))
        if ph == "afterS":
            ph2 = "digits" if dig else "afterS_ws" if ws else "none"
        elif ph == "afterS_ws":
            ph2 = "digits" if dig else "none"
        elif ph == "digits":
            ph2 = "digits" if dig else "none"
        else:
            ph2 = "none"
        if ph in ("afterS", "afterS_ws") and ph2 == "digits":
            prev = count - (1 if flag else 0)
            if started and prev < min_words:
                return DEAD
            return (k2, "digits", True, 0, True, False, False)
        if new_run:
            count = min(cap, count + 1)
        if k2 == 0:
            at_run2 = False
        elif k2 == k + 1:
            at_run2 = new_run if k == 0 else at_run
        else:
            at_run2 = False
        flag2 = flag
        if k2 == L and ph2 != "digits":
            ph2 = "afterS"
            flag2 = at_run2
        if ph2 == "none":
            flag2 = False
        return (k2, ph2, started, count, wc, at_run2, flag2)

    def accept(s):
        k, ph, started, count, in_word, at_run, flag = s
        return not started or count >= min_words

    return build((0, "none", False, 0, False, False, False), step, accept)


def repeat_prompt_hard(prompt, min_words=REPEAT_MIN_WORDS):
    """Case-SENSITIVE echo of the prompt (official: case-insensitive) followed by >= min_words word runs."""
    pat = tuple((ord(ch) for ch in prompt.strip()))
    if not pat or any((x > 127 for x in pat)):
        raise Unsupported("prompt_to_repeat must be nonempty ASCII")
    L, cap = (len(pat), min_words)

    def step(s, c):
        if isinstance(s, tuple):
            count, in_word = s
            wc = is_word_class(c)
            if wc and (not in_word):
                count = min(cap, count + 1)
            return (count, wc)
        if s == "lead":
            if is_space_class(c):
                return "lead"
            s = 0
        if c == pat[s]:
            return (0, is_word_class(c)) if s + 1 == L else s + 1
        return DEAD

    return build("lead", step, lambda s: isinstance(s, tuple) and s[0] >= cap)


def after_prefix(prompt, inner):
    """If the response starts with the (case-sensitive) echoed prompt, run `inner` on the text AFTER the echo only;
    otherwise run `inner` on the whole response. Both branches imply the official result (a match in a suffix is a
    match in the whole response), so this is sound for 'at least' keyword counts and the title."""
    pat = tuple((ord(ch) for ch in prompt.strip()))
    if not pat or any((x > 127 for x in pat)):
        raise Unsupported("prompt_to_repeat must be nonempty ASCII")
    L = len(pat)
    dead_inner = frozenset((i for i, st in enumerate(inner.states) if st is DEAD))

    def adv(q, c):
        q = int(inner.transition[q, c])
        return None if q in dead_inner else q

    def step(s, c):
        mode, k, q = s
        if mode in ("in", "all"):
            q = adv(q, c)
            return DEAD if q is None else (mode, 0, q)
        q = adv(q, c)
        if k == "lead":
            if is_space_class(c):
                return ("pre", "lead", q) if q is not None else DEAD
            k = 0
        if c == pat[k]:
            if k + 1 == L:
                return ("in", 0, inner.start)
            return ("pre", k + 1, q) if q is not None else ("pre", k + 1, -1)
        return DEAD if q is None else ("all", 0, q)

    def accept(s):
        mode, k, q = s
        return mode in ("in", "all") and bool(inner.accept[q])

    return build(("pre", "lead", inner.start), step, accept)


def _lower_set(ch):
    return PV._lower_set(ch)


def postscript_hard(marker, min_words=POSTSCRIPT_MIN_WORDS):
    """Official marker regex (lowercased) with the extra conditions: the marker starts at a word boundary and is
    followed by >= min_words word runs (to the end of the response)."""
    if marker == "P.S.":
        toks = [_lower_set("p"), _lower_set("."), None, _lower_set("s"), _lower_set(".")]
    elif marker == "P.P.S":
        toks = [
            _lower_set("p"),
            _lower_set("."),
            None,
            _lower_set("p"),
            _lower_set("."),
            None,
            _lower_set("s"),
        ]
    else:
        raise Unsupported("postscript marker " + repr(marker))
    n, cap = (len(toks), min_words)

    def closure(S):
        S = set(S)
        changed = True
        while changed:
            changed = False
            for i in list(S):
                if i < n and toks[i] is None and (i + 1 not in S):
                    S.add(i + 1)
                    changed = True
        return S

    def step(s, c):
        if s[0] == "after":
            _, count, in_word = s
            wc = is_word_class(c)
            if wc and (not in_word):
                count = min(cap, count + 1)
            return ("after", count, wc)
        _, S, prev_word = s
        wc = is_word_class(c)
        nxt = set()
        for i in S:
            if i >= n:
                continue
            tok = toks[i]
            if tok is None:
                if is_space_class(c):
                    nxt.add(i + 1)
            elif c in tok:
                nxt.add(i + 1)
        if not prev_word and c in toks[0]:
            nxt.add(1)
        nxt = closure(nxt)
        if n in nxt:
            return ("after", 0, wc)
        return ("scan", frozenset(nxt), wc)

    return build(("scan", frozenset(), False), step, lambda s: s[0] == "after" and s[1] >= cap)


def predicate(instruction_id, kw):
    kw = {k: v for k, v in (kw or {}).items() if v is not None}
    i = instruction_id
    if i == "keywords:existence":
        out = []
        for w in sorted(kw["keywords"]):
            out.append(
                keyword_whole_ascii(w, 1)
                if _ascii_word(w)
                else literal_count(w, 1, "at least", False)
            )
        return out
    if i == "keywords:forbidden_words":
        out = []
        for w in sorted(set(kw["forbidden_words"])):
            out.append(literal_count(w, 1, "less than", True))
            if _ascii_word(w) and len(w) >= FORBIDDEN_GLUE_MIN_LEN:
                out.append(forbidden_glue(w))
        return out
    if i == "keywords:frequency":
        w, n, rel = (kw["keyword"].strip(), int(kw["frequency"]), kw["relation"])
        if rel == "at least" and _ascii_word(w):
            return [keyword_whole_ascii(w, n)]
        return [literal_count(w, n, rel, False)]
    if i == "keywords:letter_frequency":
        letter, n, rel = (kw["letter"].strip(), int(kw["let_frequency"]), kw["let_relation"])
        if rel == "at least":
            return [letter_count_alpha(letter, n), triple_letter_forbid(letter)]
        return [letter_count(letter, n, rel), triple_letter_forbid(letter)]
    if i == "punctuation:no_comma":
        return [no_comma()]
    if i == "change_case:english_lowercase":
        return [case_all(False), word_count(CASE_MIN_WORDS, "at least")]
    if i == "change_case:english_capital":
        return [case_all(True), word_count(CASE_MIN_WORDS, "at least")]
    if i == "startend:quotation":
        return [quotation()]
    if i == "startend:end_checker":
        return [ends_with(kw["end_phrase"])]
    if i == "detectable_content:postscript":
        return [postscript_hard(kw["postscript_marker"].strip())]
    if i == "detectable_format:title":
        return [title_hard()]
    if i == "detectable_format:number_highlighted_sections":
        return [highlights_hard(int(kw["num_highlights"])), no_literal(HIGHLIGHT_FORBIDDEN_LITERAL)]
    if i == "detectable_content:number_placeholders":
        return [placeholders_hard(int(kw["num_placeholders"]))]
    if i == "detectable_format:number_bullet_lists":
        return [bullets(int(kw["num_bullets"])), bullet_lines()]
    if i == "length_constraints:number_paragraphs":
        return [paragraphs(int(kw["num_paragraphs"])), paragraph_blocks(), no_star_only_lines()]
    if i == "length_constraints:nth_paragraph_first_word":
        return [
            nth_paragraph_first_word(
                int(kw["num_paragraphs"]), int(kw["nth_paragraph"]), kw["first_word"]
            ),
            newline_blocks(),
        ]
    if i == "detectable_format:multiple_sections":
        sp = kw["section_spliter"].strip()
        return [sections(sp, int(kw["num_sections"])), section_blocks(sp)]
    if i == "detectable_format:constrained_response":
        return [constrained_response()]
    if i == "combination:repeat_prompt":
        return [repeat_prompt_hard(kw["prompt_to_repeat"])]
    if i == "length_constraints:number_words":
        return [word_count(int(kw["num_words"]), kw["relation"])]
    if i == "combination:two_responses":
        return [two_response_blocks()]
    raise Unsupported(i)


UNSUPPORTED_IDS = (
    "detectable_format:json_format",
    "language:response_language",
    "length_constraints:number_sentences",
    "change_case:capital_word_frequency",
)
HARDENING_ONLY_IDS = ("combination:two_responses",)
PROGRAM_IDS = ("length_constraints:number_words",)
GENERIC_IDS = ("generic:no_homoglyph", "generic:no_char_runs")
HOMOGLYPH_SKIP_IF = ("language:response_language",)


def generic_components(row):
    comps = []
    if not any((i in row["instruction_id_list"] for i in HOMOGLYPH_SKIP_IF)):
        comps.append((-1, "generic:no_homoglyph", no_homoglyph()))
    comps.append((-1, "generic:no_char_runs", no_char_runs()))
    return comps


ECHO_AWARE_IDS = ("keywords:existence", "keywords:frequency", "detectable_format:title")


def _echo_prompt(row):
    for iid, kw in zip(row["instruction_id_list"], row["kwargs"]):
        if iid == "combination:repeat_prompt":
            p = ((kw or {}).get("prompt_to_repeat") or "").strip()
            if p and p.isascii():
                return p
    return None


def _components(row, generic):
    comps, unsupported = ([], [])
    echo = _echo_prompt(row)
    for idx, (iid, kw) in enumerate(zip(row["instruction_id_list"], row["kwargs"])):
        if iid in PROGRAM_IDS:
            unsupported.append((idx, iid, "program-handled"))
            continue
        try:
            for d in predicate(iid, kw):
                if (
                    echo
                    and iid in ECHO_AWARE_IDS
                    and (
                        not (
                            iid == "keywords:frequency"
                            and (kw or {}).get("relation") == "less than"
                        )
                    )
                ):
                    d = after_prefix(echo, d)
                comps.append((idx, iid, d))
        except Unsupported as e:
            unsupported.append((idx, iid, str(e)))
        if iid in HARDENING_ONLY_IDS:
            unsupported.append((idx, iid, "program-handled (hardening-only DFA added)"))
    if generic:
        comps.extend(generic_components(row))
    return (comps, unsupported)


def _fits(comps, limit):
    try:
        product([nonempty()] + [d for _, _, d in comps], max_states=limit)
        return True
    except ValueError:
        return False


def components_for(row, generic=True, max_text_states=None):
    """Hardened components (+ generic ones). With max_text_states, the joint text product is kept under that budget by
    degrading, one step at a time, the most expensive hardened instruction back to its official base DFAs
    (hardening-only two_responses is dropped; the generic char-run rule goes first); each step is reported in
    `unsupported` as (idx, iid, 'degraded:...'). Without a budget the components are returned as is (measured
    2026-10-03: 541/541 IFEval products build, median 147 text states, 20 rows > 5k, max 56.8k)."""
    comps, unsupported = _components(row, generic)
    if max_text_states is None or _fits(comps, max_text_states):
        return (comps, unsupported)
    comps = [c for c in comps if c[1] != "generic:no_char_runs"]
    unsupported.append((-1, "generic:no_char_runs", "degraded:dropped (text budget)"))
    done = set()
    while not _fits(comps, max_text_states):
        cost = {}
        for i, iid, d in comps:
            if i >= 0 and i not in done:
                cost[i] = cost.get(i, 0) + len(d.states)
        if not cost:
            break
        idx = max(cost, key=cost.get)
        done.add(idx)
        iid, kw = (row["instruction_id_list"][idx], row["kwargs"][idx])
        comps = [c for c in comps if c[0] != idx]
        try:
            official = [] if iid in HARDENING_ONLY_IDS else PV.predicate(iid, kw)
        except Unsupported:
            official = []
        comps.extend(((idx, iid, d) for d in official))
        unsupported.append(
            (
                idx,
                iid,
                "degraded:official-only (text budget)"
                if official
                else "degraded:dropped (text budget)",
            )
        )
    return (comps, unsupported)


DEFAULT_LADDER = (None, 5000, 2000, 1000, 0)


def components_ladder(row, budgets=DEFAULT_LADDER, generic=True):
    """Yield (budget, comps, unsupported) for successively smaller text budgets. The token-graph size is only loosely
    predicted by the text product (block / word-count DFAs expand 30-140x per token context), so a compiler should
    try each level until compile_contextual_event fits its budget; budget 0 = official-only components
    (every hardened instruction degraded, generic:no_homoglyph kept)."""
    for b in budgets:
        yield (b, *components_for(row, generic=generic, max_text_states=b))

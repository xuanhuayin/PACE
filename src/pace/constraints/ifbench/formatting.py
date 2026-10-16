import string
from pace.constraints.ifbench.registry import register, positives, build, DEAD, Unsupported
from pace.constraints.base import kmp_table, kmp_advance, lower_strict
from pace.automata.alphabet import (
    N_CLASSES,
    is_space_class,
    is_word_class,
    is_lower_class,
    fold_rei,
)

ASCII_PUNCT = frozenset(map(ord, string.punctuation))
ASCII_DIGITS = frozenset(map(ord, string.digits))


def _ascii_pattern(text, what):
    pat = tuple((ord(ch) for ch in text))
    if not pat or any((x > 127 for x in pat)):
        raise Unsupported(f"{what} must be nonempty ASCII: {text!r}")
    return pat


def build_ne(start, step, accept):
    """build() with an embedded strict-mode nonempty flag: whitespace-only strings are rejected (the official
    wrapper rejects them before check_following runs, so DFAs whose language contains them must exclude them)."""

    def step2(s, c):
        inner, seen = s
        t = step(inner, c)
        if t is None or t is DEAD:
            return DEAD
        return (t, seen or not is_space_class(c))

    return build((start, False), step2, lambda s: s[1] and accept(s[0]))


def _substring_count(pat, n):
    """DFA accepting iff the exact (case-sensitive) pattern occurs >= n times, counted non-overlapping
    left-to-right (re.findall semantics)."""
    f, L = (kmp_table(pat), len(pat))

    def step(s, c):
        k, cnt = s
        k2 = kmp_advance(pat, f, k, c)
        if k2 == L:
            return (0, min(n, cnt + 1))
        return (k2, cnt)

    return build((0, 0), step, lambda s: s[1] >= n)


@register(
    "format:no_whitespace",
    note="exact: no character with str.isspace() (classes _ASCII_SPACE and 139)",
)
def no_whitespace(kw, prompt):
    return [build_ne(0, lambda s, c: DEAD if is_space_class(c) else 0, lambda s: True)]


@positives("format:no_whitespace")
def no_whitespace_pos(kw, prompt):
    return [
        "Thisisananswerwithoutanywhitespace.",
        "Water,fire,earth,air.",
        "A",
        "12345-67890!",
        "NoSpacesHere;really",
    ]


@register(
    "format:output_template",
    note='exact: the three case-sensitive substrings "My Answer:", "My Conclusion:", "Future Outlook:" each occur somewhere (order and content free)',
)
def output_template(kw, prompt):
    return [
        _substring_count(_ascii_pattern(t, "template"), 1)
        for t in ("My Answer:", "My Conclusion:", "Future Outlook:")
    ]


@positives("format:output_template")
def output_template_pos(kw, prompt):
    return [
        "My Answer: 42.\nMy Conclusion: it works.\nFuture Outlook: bright.",
        "My Answer: [yes] My Conclusion: [fine] Future Outlook: [good]",
        "Intro.\n\nMy Answer:\nSomething long here.\n\nMy Conclusion:\nDone.\n\nFuture Outlook:\nMore.\n",
        "Future Outlook: later. My Conclusion: now. My Answer: first.",
    ]


@register(
    "format:list",
    note="exact: >= 2 non-overlapping occurrences of the literal sep (re.findall(re.escape(sep)))",
)
def special_list(kw, prompt):
    return [_substring_count(_ascii_pattern(str(kw["sep"]), "sep"), 2)]


@positives("format:list")
def special_list_pos(kw, prompt):
    sep = str(kw["sep"])
    return [
        f"{sep} apples\n{sep} pears\n{sep} plums",
        f"Items: {sep}one {sep}two.",
        f"{sep}{sep}",
        f"First item {sep}\nSecond item {sep}\n",
        f"Shopping list:\n\n{sep} milk\n\n{sep} eggs\n\n{sep} bread\n",
    ]


@register(
    "format:sub-bullets",
    note='exact: value.split("*") -- every segment after a "*" (up to the next "*" or the end) contains a "-"; no "*" at all also passes (quirk: "*" anywhere, not only line starts)',
)
def sub_bullets(kw, prompt):

    def step(s, c):
        if c == 42:
            return DEAD if s == 1 else 1
        if c == 45:
            return 0
        return s

    return [build_ne(0, step, lambda s: s == 0)]


@positives("format:sub-bullets")
def sub_bullets_pos(kw, prompt):
    return [
        "* Fruit\n  - apple\n  - pear\n* Veg\n  - kale",
        "Intro line.\n* First point\n    - detail one\n    - detail two\n* Second point\n    - detail\n",
        "* A - b",
        "No bullets here at all.",
        "*\n-\n*\n\n-",
        "* Point one\n  - sub-point\n* Point two\n  - sub point\n\nClosing remark - done.",
    ]


def _options_lists(options):
    """Mirror OptionsResponseChecker.build_description: strict flag + separator choice."""
    import re

    strict = re.match("\\W*[aA]\\W*[bB]\\W*[cC]\\W*", options) is not None
    sep = "/" if "/" in options else "or" if "or" in options else ","
    return (strict, [o.strip() for o in options.split(sep)])


@register(
    "format:options",
    note='exact: strict (a/b/c-style option strings) = value is exactly one option; otherwise value.strip(ASCII punct + " ") lowered (str.lower; İ never matches) equals a stripped+lowered option. Note: newlines/tabs are NOT stripped by the checker, so a trailing "\\n" fails',
)
def options_response(kw, prompt):
    options = str(kw["options"])
    strict, opts = _options_lists(options)
    STRIP = ASCII_PUNCT | {32}
    if strict:
        pats = [_ascii_pattern(o, "option") for o in opts]

        def step(s, c):
            if s == "ACC":
                return DEAD
            nxt = frozenset(((j, k + 1) for j, k in s if pats[j][k] == c))
            if any((k == len(pats[j]) for j, k in nxt)):
                return "ACC"
            return nxt or DEAD

        return [build(frozenset(((j, 0) for j in range(len(pats)))), step, lambda s: s == "ACC")]
    targets = []
    for o in opts:
        t = o.strip(string.punctuation + " ").lower()
        if not t or any((ord(ch) > 127 for ch in t)):
            raise Unsupported(f"option {o!r} strips to empty or non-ASCII")
        targets.append(tuple(map(ord, t)))

    def step(s, c):
        started, S, done = s
        lc = lower_strict(c)
        if not started:
            if c in STRIP:
                return s
            S = frozenset(((j, 0) for j in range(len(targets))))
            started = True
        nxt, done2 = (set(), done and c in STRIP)
        for j, k in S:
            if targets[j][k] == lc:
                if k + 1 == len(targets[j]):
                    done2 = True
                else:
                    nxt.add((j, k + 1))
        if not nxt and (not done2):
            return DEAD
        return (True, frozenset(nxt), done2)

    return [build((False, frozenset(), False), step, lambda s: s[2])]


@positives("format:options")
def options_response_pos(kw, prompt):
    strict, opts = _options_lists(str(kw["options"]))
    if strict:
        return list(opts)
    out = []
    for o in opts:
        out += [o, o.upper(), o.lower(), f"{o}.", f"  {o}!! ", f'"{o}"', o.capitalize() + "..."]
    return out


@register(
    "format:newline",
    note='sufficient: after deleting ASCII punctuation, every "\\n"-line is empty or holds exactly one whitespace-separated word, and whitespace-only lines occur only before the first / after the last word (checker: #nonempty lines == #words, so e.g. a 2-word line compensated by a blank-but-spaces line also passes officially; not accepted here)',
)
def newline_words(kw, prompt):

    def step(s, c):
        seen, pend, ls = s
        if c in ASCII_PUNCT:
            return s
        if c == 10:
            if ls == "W" and seen:
                pend = True
            return (seen, pend, "E")
        if is_space_class(c):
            return (seen, pend, "W" if ls in ("E", "W") else "A")
        if pend or ls == "A":
            return DEAD
        return (True, pend, "IN")

    return [build_ne((False, False, "E"), step, lambda s: True)]


@positives("format:newline")
def newline_words_pos(kw, prompt):
    return [
        "The\nquick\nbrown\nfox.",
        "One\n\nTwo\n\nThree\n",
        "Hello,\nworld!\n",
        "single",
        "  Lead\n  ing\n  spaces  \n",
        "Roses\nare\nred.\nViolets\nare\nblue.\n\n",
        "a\nb\nc\nd\ne\nf\ng",
        "Yes.\n\n\n",
        "two words\n \nthree",
    ]


@register(
    "format:quote_unquote",
    note="exact: after deleting every literal '\"' triple (left-to-right, before whitespace removal) and then all whitespace, no two adjacent '\"' and the last character that is not an ASCII digit/punctuation(except \") is not '\"' (the curly-quote replace() calls in the checker are no-ops: both arguments are the ASCII quote)",
)
def quote_unquote(kw, prompt):
    STRIPSET = (ASCII_DIGITS | ASCII_PUNCT) - {34}

    def emit(w, c):
        """w = (last_is_quote, last_nonstrip_is_quote) over the processed stream; c already non-space."""
        lq, lns = w
        if c == 34:
            if lq:
                return DEAD
            return (True, True)
        if c in STRIPSET:
            return (False, lns)
        return (False, False)

    def feed(w, c):
        if is_space_class(c):
            return w
        return emit(w, c)

    def step(s, c):
        pend, w = s
        if pend == 1:
            if c == 34:
                return (2, w)
            w = feed(w, 39)
            if w is DEAD:
                return DEAD
            if c == 39:
                return (1, w)
            w = feed(w, c)
            return DEAD if w is DEAD else (0, w)
        if pend == 2:
            if c == 39:
                return (0, w)
            w = feed(w, 39)
            if w is DEAD:
                return DEAD
            w = feed(w, 34)
            if w is DEAD:
                return DEAD
            w = feed(w, c)
            return DEAD if w is DEAD else (0, w)
        if c == 39:
            return (1, w)
        w = feed(w, c)
        return DEAD if w is DEAD else (0, w)

    def accept(s):
        pend, w = s
        if pend == 1:
            w = feed(w, 39)
        elif pend == 2:
            w = feed(w, 39)
            if w is not DEAD:
                w = feed(w, 34)
        return w is not DEAD and (not w[1])

    return [build_ne((0, (False, False)), step, accept)]


@positives("format:quote_unquote")
def quote_unquote_pos(kw, prompt):
    return [
        'He said "carpe diem", which means seize the day.',
        'The phrase "to be or not to be" asks whether life is worth living. Then "I think" means cognition.',
        "No quotes at all here.",
        '"Alpha" is the first letter; "omega" is the last one.',
        'She wrote "hello" (a greeting) and "bye" -- a farewell, 2 words.',
        'Quote: "x". Explanation: x is a variable.\n\nQuote: "y". Explanation: y is another one.\n',
    ]


PAREN_DEPTH = 6


@register(
    "format:parentheses",
    note=f"sufficient (depth-bounded, stack <= {PAREN_DEPTH}): mirrors the checker stack exactly -- opens push, a close matching the top pops, a mismatched close (or close on empty stack) resets the stack and the max depth; accept once a matching close happens after the stack reached depth 5 since the last reset. Nesting deeper than the bound is rejected (sound)",
)
def nested_parentheses(kw, prompt):
    OPEN = {40: 41, 91: 93, 123: 125}
    CLOSE = {41, 93, 125}

    def step(s, c):
        if s == "ACC":
            return "ACC"
        stack, reached = s
        if c in OPEN:
            if len(stack) >= PAREN_DEPTH:
                return DEAD
            stack = stack + (OPEN[c],)
            return (stack, reached or len(stack) >= 5)
        if c in CLOSE:
            if stack and stack[-1] == c:
                return "ACC" if reached else (stack[:-1], reached)
            return ((), False)
        return s

    return [build(((), False), step, lambda s: s == "ACC")]


@positives("format:parentheses")
def nested_parentheses_pos(kw, prompt):
    return [
        "Here is a nest: (one [two {three (four [five] four) three} two] one).",
        "((((( deep )))))",
        "Mismatch first ) then (a(b(c(d(e(f)))))) done.",
        "([{([x])}]) and more text\n",
        "Text (a [b {c (d [e {f} e] d) c} b] a) end.",
        "{[({[ ]})]}",
        "(((((((seven deep)))))))",
    ]


QUOTE_DEPTH = 24


@register(
    "format:quotes",
    note=f"""sufficient (depth-bounded, stack <= {QUOTE_DEPTH}): mirrors the checker exactly -- a quote char equal to the stack top pops, otherwise any '"' or "'" pushes (the stack therefore alternates and is determined by its bottom char + depth); the max depth is never reset; accept when a pop brings the depth to max_depth - 3 or lower. Apostrophes in words count as quotes""",
)
def nested_quotes(kw, prompt):

    def top(stack):
        bottom, depth = stack
        return bottom if depth % 2 == 1 else 34 if bottom == 39 else 39

    def step(s, c):
        if s == "ACC":
            return "ACC"
        (bottom, depth), reached = s
        if depth and c == top((bottom, depth)):
            depth -= 1
            if reached - depth >= 3:
                return "ACC"
            return ((bottom if depth else 0, depth), reached)
        if c in (34, 39):
            if depth >= QUOTE_DEPTH:
                return DEAD
            if depth == 0:
                bottom = c
            depth += 1
            return ((bottom, depth), max(reached, depth))
        return s

    return [build(((0, 0), 0), step, lambda s: s == "ACC")]


@positives("format:quotes")
def nested_quotes_pos(kw, prompt):
    return [
        'He said "she told me \'he whispered "run" to her\' yesterday" and left.',
        '"\'"x"\'"',
        "She said 'he wrote \"they shouted 'go' loudly\" in the note' to me.",
        'Nested: "level one \'level two "level three" two\' one".\n\nThat is all.',
        'Do not worry: "it is \'fine "really" fine\' here" he said.',
        'Level one begins "here, level two \'starts, level three "sits" ends\' and" closes.\n',
    ]


@register(
    "format:thesis",
    note='exact: i0 = first "<i>" if any else first "<em>"; e = first "</i>" after i0 if any else first "</em>"; value[i0+3:e].strip() and value[e+4:].strip() nonempty (quirks mirrored: the thesis slice always starts 3 chars in, so an "<em>" thesis is never empty; after "</em>" the text slice starts at its ">" so it is never empty)',
)
def italics_thesis(kw, prompt):
    LT, GT, SL, I, E, M = (60, 62, 47, 105, 101, 109)
    TAGS = {"<i": I, "<e": E, "<em": M, "</i": I, "</e": E, "</em": M}

    def adv(m, c):
        """Tag matcher. Returns (new_matcher, event); event: 'pend' (c absorbed into a tag prefix, not content
        yet), 'plain' (c is an ordinary char), 'flush' (the pending prefix chars become ordinary content; c is
        ordinary too unless it is '<', which starts a new prefix), or a completed tag name."""
        if c == LT:
            return ("<", "flush" if m != "r" else "pend")
        if m == "r":
            return ("r", "plain")
        if m == "<":
            if c == I:
                return ("<i", "pend")
            if c == E:
                return ("<e", "pend")
            if c == SL:
                return ("</", "pend")
            return ("r", "flush")
        if m in ("<i", "<em", "</i", "</em"):
            if c == GT:
                return ("r", m + ">")
            return ("r", "flush")
        if m in ("<e", "</e"):
            if c == M:
                return (m + "m", "pend")
            return ("r", "flush")
        if m == "</":
            if c == I:
                return ("</i", "pend")
            if c == E:
                return ("</e", "pend")
            return ("r", "flush")
        raise AssertionError(m)

    def step(s, c):
        m, ph = s
        m2, ev = adv(m, c)
        plain = ev in ("pend", "plain", "flush")
        content = ev == "flush" or (ev == "plain" and (not is_space_class(c)))
        phase = ph[0]
        if phase == 0:
            if ev == "<i>":
                return (m2, ("A1", False, None))
            if ev == "<em>":
                return (m2, ("B", False, False, False))
            return (m2, ph)
        if phase == "B":
            _, iend, text, emend = ph
            if ev == "<i>":
                return (m2, ("A1", False, None))
            if ev == "</i>":
                return (m2, ("B", True, True, emend)) if iend else (m2, ("B", True, False, emend))
            if ev == "</em>":
                return (m2, ("B", iend, True, emend)) if iend else (m2, ("B", iend, text, True))
            if ev == "<em>":
                return (m2, ("B", iend, text or iend, emend))
            return (m2, ("B", iend, text or (iend and content), emend))
        if phase == "A1":
            _, cont, emend = ph
            if ev == "</i>":
                return (m2, ("A2", cont, False))
            if ev == "</em>":
                return (m2, ("A1", True, cont if emend is None else emend))
            if ev in ("<i>", "<em>"):
                return (m2, ("A1", True, emend))
            return (m2, ("A1", cont or content, emend))
        _, thesis, text = ph
        if not plain:
            return (m2, ("A2", thesis, True))
        return (m2, ("A2", thesis, text or content))

    def accept(s):
        m, ph = s
        pend = m != "r"
        phase = ph[0]
        if phase == "A2":
            return ph[1] and (ph[2] or pend)
        if phase == "A1":
            return ph[2] is True
        if phase == "B":
            _, iend, text, emend = ph
            return text or pend if iend else emend
        return False

    return [build(("r", (0,)), step, accept)]


@positives("format:thesis")
def italics_thesis_pos(kw, prompt):
    return [
        "<i>Cities must invest in transit.</i>\nBuses and trains cut emissions and congestion.",
        "Section 1\n<em>Reading daily improves focus.</em> Studies show a measurable effect.\n\nSection 2\n<em>Sleep matters.</em> It consolidates memory.",
        "<i>Thesis one.</i> Body one.\n\n<i>Thesis two.</i> Body two.\n",
        "Intro <i> x </i> y",
        "<i>A thesis</i>\n\n\nFollowed by text after blank lines.\n",
        "<em>Only em tags here.</em>",
    ]


INDENT_CAP = 30


@register(
    "format:line_indent",
    note=f'sufficient (indent <= {INDENT_CAP} spaces): every "\\n"-line has strictly more leading U+0020 spaces than the previous non-blank line, and no two consecutive blank (whitespace-only) lines (quirk: the checker removes blank lines while iterating, so the second of two adjacent blank lines survives with its own indent; a trailing "\\n\\n" therefore fails). Indents beyond the cap are rejected',
)
def indent_stairs(kw, prompt):

    def step(s, c):
        ph = s[0]
        if ph == "C":
            return ("L", s[1], 0, False) if c == 10 else s
        _, prev, cur, flag = s
        if c == 10:
            return DEAD if flag else ("L", prev, 0, True)
        if c == 32 and ph == "L":
            return DEAD if cur >= INDENT_CAP else ("L", prev, cur + 1, flag)
        if is_space_class(c):
            return ("B", prev, cur, flag)
        return ("C", cur) if cur > prev else DEAD

    def accept(s):
        return s[0] == "C" or not s[3]

    return [build_ne(("L", -1, 0, False), step, accept)]


@positives("format:line_indent")
def indent_stairs_pos(kw, prompt):
    return [
        "Step one\n Step two\n  Step three\n   Step four\n",
        "A\n  B\n    C\n      D\n        E",
        "one\n one\n  one\n   one\n    one\n     one\n      one\n       one",
        "First line\n\n    Second, after a blank line.\n\n        Third.\n",
        "\nLeading newline, then deeper lines.\n  Two spaces.\n\n     Five spaces, after one blank line.\n",
        "x\n y\n  z",
        "a\n" + " " * 35 + "b",
    ]


ASCII_WS = frozenset(map(ord, string.whitespace))
REALIGN_ASCII = frozenset(map(ord, "\"')]}"))


@register(
    "format:no_bullets_bullets",
    note='sufficient: lines = value.split("\\n"); a prefix of non-blank prose lines (optionally ended by ONE blank line), then >= 2 lines whose first non-space char is "*", nothing after (no blank line, no trailing "\\n": the empty last line makes the checker return False). Sentences are lower-bounded by 1 per non-blank prose line plus one per [?!] followed by whitespace containing an ASCII whitespace char and then a non-space char outside "\'\\")]}" and non-ASCII punctuation (verified sound against nltk Punkt by fuzzing); the count must reach 2 before the first bullet. Period-only sentence splits inside one line are not counted (Punkt abbreviation/collocation logic is not regular)',
)
def some_bullet_points(kw, prompt):

    def step(s, c):
        phase, sc, bc, line = s
        if c == 10:
            if line == "S":
                return ("PB", sc, bc, "S") if phase == "P" else DEAD
            if line == "BUL":
                return DEAD if sc < 2 else ("BL", sc, min(2, bc + 1), "S")
            return ("P", min(2, sc + 1), bc, "S")
        if line == "BUL":
            return s
        if line == "S":
            if is_space_class(c):
                return s
            if c == 42:
                return (phase, sc, bc, "BUL")
            if phase != "P":
                return DEAD
            return (phase, sc, bc, ("T", "q" if c in (63, 33) else "n"))
        q = line[1]
        if c in (63, 33):
            return (phase, sc, bc, ("T", "q"))
        if is_space_class(c):
            if q == "n":
                return s
            return (phase, sc, bc, ("T", "qa" if c in ASCII_WS or q == "qa" else "qw"))
        if q == "qa" and c not in REALIGN_ASCII and (c != 132):
            sc = min(2, sc + 1)
        return (phase, sc, bc, ("T", "n"))

    def accept(s):
        phase, sc, bc, line = s
        return line == "BUL" and sc >= 2 and (bc + 1 >= 2)

    return [build(("P", 0, 0, "S"), step, accept)]


@positives("format:no_bullets_bullets")
def some_bullet_points_pos(kw, prompt):
    return [
        "This is the first sentence.\nThis is the second sentence.\n* bullet one\n* bullet two",
        "Is it ready? Yes it is! Great.\n* alpha\n* beta\n* gamma",
        "First sentence here.\nSecond sentence here.\n\n* one\n  * two\n* three",
        "Line one. Line two. Line three.\nAnother line.\n* a\n* b",
        "One\nTwo\nThree\n* x\n* y",
        "Really? Really.\n* p\n* q",
        "First sentence. Second sentence.\n* a\n* b",
    ]


TITLE_FORBID_I = (
    "cannot",
    "gimme",
    "gonna",
    "gotta",
    "lemme",
    "wanna",
    "d'ye",
    "more'n",
    "'tis",
    "'twas",
)
TITLE_FORBID_CS = "n't"


@register(
    "format:title_case",
    note='sufficient: no token may start with a lowercase letter, where a token start is any position preceded by a non-\\w char (or the start), except a letter right after an ASCII apostrophe that follows a \\w char (Treebank keeps "O\'Brien" whole and splits "It\'s" as "\'s", which is skipped); the Treebank contraction splits that create lowercase-initial tokens inside words are excluded by forbidding the substrings cannot/gimme/gonna/gotta/lemme/wanna/d\'ye/more\'n/\'tis/\'twas (case-insensitive) and "n\'t". Lowercase-initial tokens that the checker tolerates (mixed-case rest like "iPhone", "a1") are not accepted; non-ASCII lowercase letters (class 134, ı, ſ) are treated as lowercase alpha (conservative)',
)
def title_case(kw, prompt):
    pats = [tuple(map(ord, p)) for p in TITLE_FORBID_I]
    prefixes = {p[:k] for p in pats for k in range(len(p) + 1)}
    full = set(pats)
    cs = tuple(map(ord, TITLE_FORBID_CS))
    csf = kmp_table(cs)

    def step(s, c):
        prev, ac, k = s
        if prev == "o" and is_lower_class(c) and is_word_class(c):
            return DEAD
        cand = ac + (fold_rei(c),)
        while cand and cand not in prefixes:
            cand = cand[1:]
        if cand in full:
            return DEAD
        k2 = kmp_advance(cs, csf, k, c)
        if k2 == len(cs):
            return DEAD
        if is_word_class(c):
            prev2 = "w"
        elif c == 39 and prev == "w":
            prev2 = "a"
        else:
            prev2 = "o"
        return (prev2, cand, k2)

    return [build_ne(("o", (), 0), step, lambda s: True)]


@positives("format:title_case")
def title_case_pos(kw, prompt):
    return [
        "The Quick Brown Fox Jumps Over The Lazy Dog.",
        "Why Is The Sky Blue? Because Of Rayleigh Scattering!",
        'Title Case Everywhere:\n\nEvery Word Starts With A Capital Letter, Even "Quoted" Ones (And Bracketed Ones).\n',
        "NASA Launched 3 Rockets In 2024; All Of Them Reached Orbit.",
        "O'Brien's Pub Is Open -- It's Great.",
        "Alpha\nBeta\nGamma\n",
        "A Single Letter I Am Fine.",
        "Well-Known Facts: X-Ray Vision Is Fiction.",
        "The iPhone Is Popular.",
    ]

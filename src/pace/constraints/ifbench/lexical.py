import re
import string
from pace.constraints.ifbench.registry import (
    register,
    positives,
    build,
    DEAD,
    Unsupported,
    nonempty,
)
from pace.constraints.base import literal_count
from pace.automata.alphabet import is_space_class, is_word_class, is_decimal_class, fold_lower

PUNCT = frozenset(map(ord, string.punctuation))
CONSONANTS = frozenset(map(ord, "bcdfghjklmnpqrstvwxyz"))
VOWELS = frozenset(map(ord, "aeiou"))
PRIMES = frozenset(
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
)
CONJ = ("and", "but", "for", "nor", "or", "so", "yet")


def _ascii_letter_lower(c):
    """Lowercase ASCII letter code for an ASCII letter class, else None (non-ASCII never lowers to one of
    these words' letters: only İ->i̇ and K->k reach ASCII, and we never rely on them here)."""
    if 65 <= c <= 90:
        return c + 32
    if 97 <= c <= 122:
        return c
    return None


@register(
    "words:consonants",
    note="exact: every whitespace token of value.lower() has two adjacent ASCII consonants (K->k counts, İ->i̇ does not); nonempty added",
)
def consonants(kw, prompt):

    def step(s, c):
        inw, prev, ok = s
        if is_space_class(c):
            if inw and (not ok):
                return DEAD
            return (False, False, False)
        is_c = fold_lower(c) in CONSONANTS
        return (True, is_c, ok or (prev and is_c))

    return [build((False, False, False), step, lambda s: not s[0] or s[2]), nonempty()]


@positives("words:consonants")
def consonants_pos(kw, prompt):
    return [
        "Strong winds swept across the plains, bending every tall tree.",
        "The quick brown wolf jumps past the sleepy dogs.",
        "Twelve bright stars glittered. Nightfall brought cold winds; everybody stayed inside.",
        "FIRST THINGS FIRST:\nplanting trees helps towns breathe.\n\nSecond, strong roots hold hillsides firmly.",
        "Thanks!",
        "  Clever students write short, crisp drafts.  \n",
    ]


@register(
    "words:vowel",
    note="exact: value.strip() contains no '\\n' and <= 3 distinct ASCII vowels after lower() (İ->i counts); nonempty added",
)
def vowel(kw, prompt):

    def step(s, c):
        phase, vs = s
        if c == 10:
            return (0 if phase == 0 else 2, vs)
        if is_space_class(c):
            return s
        if phase == 2:
            return DEAD
        v = fold_lower(c)
        if v in VOWELS:
            vs = vs | 1 << "aeiou".index(chr(v))
            if bin(vs).count("1") > 3:
                return DEAD
        return (1, vs)

    return [build((0, 0), step, lambda s: True), nonempty()]


@positives("words:vowel")
def vowel_pos(kw, prompt):
    return [
        "A small bird sat on a log. It did not stop singing, so I sat and did nothing.",
        "Henry went home. He did not stop; the storm kept rolling in.",
        "Eli sat beside the lake, reading. Rain fell again; it was late.",
        "  Big dogs bark at night, and cats hiss back.\n",
        "STOP! LOOK AT THAT BIG CAT.",
        "Tom took his dog for a long walk; it was hot, so both got a drink.",
    ]


@register(
    "words:no_consecutive",
    note='exact on ASCII/İ/K/ı/ſ word starts (lower(), ASCII punctuation removed BEFORE split(), so "a.b" is one word); two consecutive words both starting with other non-ASCII chars -> reject (conservative); nonempty added',
)
def no_consecutive(kw, prompt):

    def key(c):
        if c < 128 or c in (128, 131):
            return fold_lower(c)
        if c in (129, 130):
            return c
        return "NA"

    def step(s, c):
        inw, last = s
        if c in PUNCT:
            return s
        if is_space_class(c):
            return (False, last)
        if inw:
            return s
        k = key(c)
        if last is not None and k == last:
            return DEAD
        return (True, k)

    return [build((False, None), step, lambda s: True), nonempty()]


@positives("words:no_consecutive")
def no_consecutive_pos(kw, prompt):
    return [
        "The quick brown fox jumps over a lazy dog.",
        "My neighbor bakes fresh bread every Sunday; the smell drifts across our street.",
        "First, gather your tools.\nNext, measure twice and cut once.\nFinally, sand the edges smooth.",
        "In 2024, nearly 300 teams joined; most arrived early.",
        "Hello world",
        '"Wait," he said. "Nobody told me about this plan!"',
    ]


@register(
    "words:alphabet",
    note='exact: ASCII punctuation removed BEFORE split() ("a.b" is one word); word starts must follow a..z cyclically; first word may start with K (->k) but not İ (lower() is 2 chars), later words may start with İ (->i) or K; digits/other -> fail; needs >= 1 word',
)
def alphabet(kw, prompt):

    def first_key(c, first_word):
        k = _ascii_letter_lower(c)
        if k is not None:
            return k
        if c == 131:
            return 107
        if c == 128 and (not first_word):
            return 105
        return None

    def step(s, c):
        inw, exp = s
        if c in PUNCT:
            return s
        if is_space_class(c):
            return (False, exp)
        if inw:
            return s
        k = first_key(c, exp is None)
        if k is None or (exp is not None and k != exp):
            return DEAD
        return (True, 97 + (k - 97 + 1) % 26)

    return [build((False, None), step, lambda s: s[1] is not None)]


@positives("words:alphabet")
def alphabet_pos(kw, prompt):
    return [
        "A big cat danced elegantly, finding great happiness in jumping.",
        "Xavier yelled zealously. Amy brought cake; Dan ended frantically.",
        "Kind lions make noise - often playing quietly.",
        "QUIET RAIN SOFTLY TAPPED UPON VAST WINDOWS.",
        "Many\nnights\nowls\nperch\nquietly.",
        "Under violet winds, xenial yaks zigzag around brambles.",
    ]


@register(
    "words:prime_lengths",
    note="exact: ASCII punctuation removed, split(); every token length (code points) in the prime set up to 97; nonempty added",
)
def prime_lengths(kw, prompt):

    def step(s, c):
        if is_space_class(c):
            if s and s not in PRIMES:
                return DEAD
            return 0
        if c in PUNCT:
            return s
        return min(98, s + 1)

    return [build(0, step, lambda s: s == 0 or s in PRIMES), nonempty()]


@positives("words:prime_lengths")
def prime_lengths_pos(kw, prompt):
    return [
        "The cat sat on the mat.",
        "Every night, small birds chirp; we adore their happy songs.",
        "Prime numbers: 11, 13, 17.",
        "Hi there!\nWe are all happy today.\nThe sun is out.",
        "I'm happy.",
        "OLD ships still go far; sailors never sleep.",
    ]


@register(
    "count:numbers",
    note="exact: ASCII punctuation removed (so 3.14 / 1,000 / 10-20 are ONE number each), then count maximal runs of \\d (ASCII digits + class 136) == N",
)
def numbers(kw, prompt):
    n = int(kw["N"])
    cap = n + 1

    def step(s, c):
        inr, cnt = s
        if c in PUNCT:
            return s
        if is_decimal_class(c):
            return (True, cnt if inr else min(cap, cnt + 1))
        return (False, cnt)

    return [build((False, 0), step, lambda s: s[1] == n)]


@positives("count:numbers")
def numbers_pos(kw, prompt):
    n = int(kw["N"])
    fill = ["we counted", "then saw", "and later", "plus about", "followed by", "and finally"]
    vals = [str(3 + 7 * i % 50) for i in range(n)]
    v1 = (
        "On the trip "
        + ", ".join((f"{fill[i % len(fill)]} {vals[i]} birds" for i in range(n)))
        + ". What a day!"
    )
    fancy = ["3.5", "1,000", "10-20", "2024/05", "7th", "v2", "12:30", "$45", "99%", "0.001"]
    v2 = (
        "Figures: "
        + "; ".join((fancy[i % len(fancy)] for i in range(n)))
        + ". These are all the numbers."
    )
    v3 = "Shopping list\n" + "\n".join((f"{i + 1}. item" for i in range(n))) + "\nDone."
    return [v1, v2, v3]


@register(
    "count:punctuation",
    note="exact except lone '‽': needs '?!' or '!?', >= 2 '!' and >= 2 '?' (the first '?!' else first '!?' pair is deleted before the cover test), and . , ; :",
)
def punctuation(kw, prompt):
    BITS = {46: 1, 44: 2, 59: 4, 58: 8}

    def step(s, c):
        ib, prev, ne, nq, seen = s
        if c == 33:
            ib, ne, prev = (ib or prev == 63, min(2, ne + 1), 33)
        elif c == 63:
            ib, nq, prev = (ib or prev == 33, min(2, nq + 1), 63)
        else:
            prev = 0
            seen |= BITS.get(c, 0)
        if ib:
            prev = 0
        return (ib, prev, ne, nq, seen)

    return [
        build(
            (False, 0, 0, 0, 0), step, lambda s: s[0] and s[2] == 2 and (s[3] == 2) and (s[4] == 15)
        )
    ]


@positives("count:punctuation")
def punctuation_pos(kw, prompt):
    return [
        "Wait, what?! You finished already? That is amazing! Here is the plan: first, rest; then, celebrate.",
        "Really!? No way! Who told you? Listen: we leave at dawn, pack light; bring water.",
        "Dear team,\n\nDid anyone see the report?! I did not; neither did Sam. Why not? Because: nobody sent it!\n",
        "Stop!? Yes, stop. Why? Because it is late; also: you are tired, and so am I!",
    ]


@register(
    "count:conjunctions",
    note="sufficient: >= n distinct normalised conjunctions (token.strip(punct).lower() in {and,but,for,nor,or,so,yet}); checker counts distinct RAW tokens, so this implies official True; gap = casing/punctuation variants of one word",
)
def conjunctions(kw, prompt):
    n = int(kw["small_n"])
    if n <= 0:
        return [build(0, lambda s, c: 0, lambda s: True)]
    prefixes = {w[:i] for w in CONJ for i in range(1, len(w) + 1)}
    ACC = "ACC"

    def finalize(tok, seen):
        if tok[0] == "P" and tok[1] in CONJ or tok[0] == "T":
            seen = seen | frozenset([tok[1]])
        return seen

    def step(s, c):
        if s == ACC:
            return ACC
        tok, seen = s
        if is_space_class(c):
            seen = finalize(tok, seen)
            return ACC if len(seen) >= n else (("B",), seen)
        if tok[0] == "F":
            return s
        is_p = c in PUNCT
        letter = _ascii_letter_lower(c)
        letter = chr(letter) if letter is not None else None
        if tok[0] == "B":
            if is_p:
                return s
            return (("P", letter), seen) if letter in prefixes else (("F",), seen)
        if tok[0] == "P":
            if letter is not None:
                p = tok[1] + letter
                return (("P", p), seen) if p in prefixes else (("F",), seen)
            if is_p and tok[1] in CONJ:
                return (("T", tok[1]), seen)
            return (("F",), seen)
        if is_p:
            return s
        return (("F",), seen)

    def accept(s):
        return s == ACC or len(finalize(*s)) >= n

    return [build((("B",), frozenset()), step, accept)]


@positives("count:conjunctions")
def conjunctions_pos(kw, prompt):
    n = int(kw["small_n"])
    pieces = {
        "and": "bread and butter",
        "but": "cheap but good",
        "or": "tea or coffee",
        "so": "so we stayed",
        "yet": "yet nobody left",
        "for": "for the whole night",
        "nor": "nor did we sleep",
    }
    order = ["and", "but", "or", "so", "yet", "for", "nor"]
    out = [
        "I wanted tea, but there was none, so I made coffee and toast, yet I was still hungry, for breakfast had been small.",
        "Neither the rain nor the wind stopped us, and we kept walking; you could rest or push on, but we chose to push on, so we arrived early.",
        "Plan A: leave at dawn and drive north.\nPlan B: wait for the storm to pass, or take the train.\n\nWe liked neither, yet we had to choose, so we flipped a coin.",
        "It was " + ", ".join((pieces[w] for w in order[: min(n, 7)])) + ".",
        "AND YET, OR SO they say, BUT who knows? Nor I, for one.",
    ]
    if n <= 2:
        out.append("And then, and then, and again.")
    return out


@register(
    "count:words_japanese",
    note="sufficient (degenerate): class alphabet cannot certify Japanese, so every N-th whitespace token must be exempt: all ASCII punctuation, or a decimal number (ASCII digits/class 136) wrapped in optional ASCII punctuation; nonempty added",
)
def words_japanese(kw, prompt):
    N = int(kw["N"])
    if N <= 0:
        raise Unsupported("N must be positive (checker divides by N)")

    def step(s, c):
        k, tok = s
        if is_space_class(c):
            return s if tok is None else ((k + 1) % N, None)
        if tok is None:
            if (k + 1) % N:
                return (k, "X")
            tok = "P"
        if tok == "X":
            return s
        is_p, is_d = (c in PUNCT, is_decimal_class(c))
        if tok == "P":
            nt = "P" if is_p else "D" if is_d else None
        elif tok == "D":
            nt = "D" if is_d else "Q" if is_p else None
        else:
            nt = "Q" if is_p else None
        return DEAD if nt is None else (k, nt)

    return [build((0, None), step, lambda s: True), nonempty()]


_JP = [
    "東京",
    "寿司",
    "ありがとう",
    "桜",
    "猫",
    "日本語",
    "勉強",
    "図書館",
    "友達",
    "電車",
    "カタカナ",
    "山",
]
_FILL = "the quiet morning light spread over the old harbor while fishermen prepared their small boats and gulls circled above the market stalls where vendors arranged fresh fruit before the first customers arrived from the nearby villages carrying baskets".split()


def _jp_text(N, length, nth, punct_every=9, newline_every=0):
    toks = []
    for i in range(length):
        t = nth(i) if (i + 1) % N == 0 else _FILL[i % len(_FILL)]
        if punct_every and (i + 1) % punct_every == 0:
            t += ","
        toks.append(t)
    toks[-1] = toks[-1].rstrip(",") + "."
    if newline_every:
        return "\n".join(
            (" ".join(toks[j : j + newline_every]) for j in range(0, len(toks), newline_every))
        )
    return " ".join(toks)


@positives("count:words_japanese")
def words_japanese_pos(kw, prompt):
    N = int(kw["N"])
    L = max(3 * N + 2, 12)
    return [
        _jp_text(N, L, lambda i: _JP[i % len(_JP)]),
        _jp_text(N, L, lambda i: f"{_JP[i % len(_JP)]}.", newline_every=11),
        _jp_text(N, L, lambda i: str(10 + i)),
        _jp_text(N, L, lambda i: ["(42)", "7.", "--", "[3000]", '"12"'][i % 5], punct_every=0),
        _jp_text(N, L, lambda i: _JP[i % len(_JP)] if i % 2 else str(i)),
    ]


_KW_KEYS = ("keyword1", "keyword2", "keyword3", "keyword4", "keyword5")
_KW_COUNTS = (1, 2, 3, 5, 7)


@register(
    "count:keywords_multiple",
    note="exact: one literal_count DFA per keyword (\\b..\\b, re.I, exact count 1/2/3/5/7); keywords outside [A-Za-z0-9_]+ -> Unsupported",
)
def keywords_multiple(kw, prompt):
    out = []
    for key, cnt in zip(_KW_KEYS, _KW_COUNTS):
        if key not in kw:
            raise Unsupported(f"{key} missing: checker draws a random keyword")
        word = str(kw[key]).strip()
        if not re.fullmatch("[A-Za-z0-9_]+", word):
            raise Unsupported(
                f"keyword {word!r} is not a pure \\w ASCII word; \\b semantics differ"
            )
        out.append(literal_count(word, cnt, "exact", True))
    return out


@positives("count:keywords_multiple")
def keywords_multiple_pos(kw, prompt):
    ks = [str(kw[k]).strip() for k in _KW_KEYS]
    t = {
        1: "The {k} stood alone in the dark.",
        2: "First the {k}, then another {k}.",
        3: "One {k}, two {k}, three {k}: that was all.",
        5: "A {k} here, a {k} there, a {k} above, a {k} below, and a {k} in the middle.",
        7: "Count them: {k}, {k}, {k}, {k}, {k}, {k}, {k}.",
    }
    v1 = " ".join((t[c].format(k=k) for k, c in zip(ks, _KW_COUNTS)))
    v2 = "\n".join((t[c].format(k=k.capitalize()) for k, c in zip(ks, _KW_COUNTS)))
    v3 = "\n\n".join((t[c].format(k=f'"{k.upper()}"') for k, c in zip(ks, _KW_COUNTS)))
    v4 = (
        " ".join((t[c].format(k=k) for k, c in zip(ks, _KW_COUNTS)))
        + f" (Not {ks[0]}s, {ks[1]}ish, or {ks[4]}ing.)"
    )
    out = []
    for v in (v1, v2, v3, v4):
        if all(
            (
                len(re.findall("\\b%s\\b" % re.escape(k), v, re.I)) == c
                for k, c in zip(ks, _KW_COUNTS)
            )
        ):
            out.append(v)
    return out


_NLTK_W = frozenset([128, 129, 130, 131, 134, 135, 136, 137, 138, 140])
_NLTK_AMBIG = frozenset([132, 133, 141])


def _nltk_word_kind(c):
    if c < 128:
        return "W" if is_word_class(c) else "N"
    if c in _NLTK_W:
        return "W"
    if c in _NLTK_AMBIG:
        return "U"
    return "N"


@register(
    "count:word_count_range",
    note='exact for text without classes 132/133; nltk RegexpTokenizer(r"\\w+") uses the `regex` engine (\\w = Alphabetic|Mark|Nd|Pc|Join_Control); ambiguous classes 132/133 are simulated nondeterministically (all readings must be in range, count spread capped at 2) -> sound; counter capped at max_words+1',
)
def word_count_range(kw, prompt):
    lo, hi = (int(kw["min_words"]), int(kw["max_words"]))
    if hi < 0:
        raise Unsupported("negative max_words: checker draws a random range")
    spread = 2 if hi <= 120 else 1

    def step(S, c):
        kind = _nltk_word_kind(c)
        nxt = set()
        for cnt, inw in S:
            if kind in ("W", "U"):
                nxt.add((cnt if inw else cnt + 1, True))
            if kind in ("N", "U"):
                nxt.add((cnt, False))
        counts = [x[0] for x in nxt]
        if max(counts) > hi or max(counts) - min(counts) > spread:
            return DEAD
        return frozenset(nxt)

    return [build(frozenset([(0, False)]), step, lambda S: all((lo <= cnt <= hi for cnt, _ in S)))]


_POOL = "the river bends slowly past old mills and quiet farms where children wave at passing boats while herons stand still in shallow water waiting for silver fish to drift by under a pale evening sky".split()


def _wc_text(m, variant):
    words = [_POOL[i % len(_POOL)] for i in range(m)]
    if variant == 1:
        words = [w.capitalize() if i % 8 == 0 else w for i, w in enumerate(words)]
        words = ["2024" if i % 13 == 12 else w for i, w in enumerate(words)]
    out = []
    for i, w in enumerate(words):
        sep = " "
        if (i + 1) % 8 == 0:
            sep = ". " if variant == 0 else ".\n" if (i + 1) % 24 == 0 else ", "
        out.append(w + (sep if i + 1 < m else ""))
    text = "".join(out).rstrip() + "."
    return "Summary:\n\n" + text + "\n" if variant == 2 else text


@positives("count:word_count_range")
def word_count_range_pos(kw, prompt):
    lo, hi = (int(kw["min_words"]), int(kw["max_words"]))
    out = [_wc_text(lo, 0), _wc_text(hi, 1), _wc_text((lo + hi) // 2, 2)]
    if hi > lo:
        out.append(_wc_text(lo + 1, 1))
    return out

from pace.paths import PROJECT_ROOT
import inspect
import os
import re
import sys
import time
import unicodedata
import zlib
from collections import Counter
from pathlib import Path

ROOT = PROJECT_ROOT


from pace.models.tess.engine import certify

WORD_RE = re.compile("\\w+")
ALPHA_RE = re.compile("[^\\W\\d_]+")
SENT_RE = re.compile("[.!?\\u0964\\u0965\\u06d4\\u3002]+(?=\\s|$)")
OPTIONS = ("My answer is yes.", "My answer is no.", "My answer is maybe.")
BLOCK_UNMET = dict(
    paragraphs="number_paragraphs",
    bullets="number_bullet_lists",
    sections="multiple_sections",
    nthpara="nth_paragraph_first_word",
    two="two_responses",
)
STOP = set(
    "the a an and or of to in is are was were be been it that this for on with as at by from you your i we they he she his her its our their not but if so than then there here what which who will can".split()
)
TITLE_PLACEHOLDERS = {
    "title",
    "your title",
    "insert title",
    "title here",
    "your title here",
    "insert title here",
    "poem of joy",
    "untitled",
    "heading",
    "name",
    "none",
    "n/a",
    "tbd",
    "todo",
    "!--",
    "--",
}
FILLER_LINE = re.compile(
    "^\\s*(?:[*\\-]\\s*)?(?:content|text|placeholder|bullet point\\s*\\d*|item\\s*\\d*|point\\s*\\d*|lorem ipsum|n/a|tbd|todo|xxx+|\\.\\.\\.)\\.?\\s*$",
    re.I,
)
SLOT_OF = {
    "detectable_content:number_placeholders": "placeholders",
    "detectable_format:title": "title",
    "detectable_format:number_highlighted_sections": "highlights",
    "length_constraints:number_paragraphs": "paragraphs",
    "detectable_format:multiple_sections": "sections",
    "combination:two_responses": "two",
    "detectable_format:number_bullet_lists": "bullets",
    "length_constraints:nth_paragraph_first_word": "nthpara",
    "combination:repeat_prompt": "repeat_prompt",
    "detectable_content:postscript": "postscript",
}
BLOCK_SLOTS = ("paragraphs", "sections", "two", "bullets", "nthpara")
PH_BAD = re.compile(
    "^\\s*(?:placeholder|insert|note|end|source|citation needed|x+|\\.+|…|_+|-+|\\?+)(?:\\b|$)",
    re.I,
)
TRIPLE_OK = re.compile(
    "^(?:i{3}|x{3}|w{3}|z{3,}|m{3,}|h?m{3,}|s{3,}h+|a+h{3,}|o{3,}h*|a{3,}|hah(?:a)+|e{3,}k|br{3,}|g{3,}|r{3,}|u{3,}|y{3,}|no{3,}|lo{3,}|coo{3,}|ii{2}i)$",
    re.I,
)
CYR_GREEK = re.compile("[Ͱ-ϿЀ-ӿ]")
LATIN = re.compile("[A-Za-z]")
BLOCK_MIN_WORDS, BULLET_MIN_WORDS, POSTSCRIPT_MIN_WORDS, REPEAT_MIN_WORDS = (8, 4, 4, 20)
try:
    import pace.constraints.ifeval as _PH

    BLOCK_MIN_WORDS, BULLET_MIN_WORDS = (int(_PH.BLOCK_MIN_WORDS), int(_PH.BULLET_MIN_WORDS))
    POSTSCRIPT_MIN_WORDS, REPEAT_MIN_WORDS = (
        int(_PH.POSTSCRIPT_MIN_WORDS),
        int(_PH.REPEAT_MIN_WORDS),
    )
except Exception:
    _PH = None


def count_words(t):
    return len(WORD_RE.findall(t))


def count_sent(t):
    """Sentence count that ignores list numbers ('2.' at a line start)."""
    t = re.sub("(?m)^\\s*\\d+[.)]\\s", "", t)
    return max(1, len(SENT_RE.findall(t))) if t.strip() else 0


def norm(s):
    return " ".join(WORD_RE.findall(s.lower()))


def _jac(na, nb, jac=0.8):
    if na == nb:
        return True
    sa, sb = (set(na.split()), set(nb.split()))
    if min(len(sa), len(sb)) < 5:
        return False
    return len(sa & sb) / len(sa | sb) >= jac


def near_dup(a, b, jac=0.8):
    na, nb = (norm(a), norm(b))
    return bool(na and nb) and _jac(na, nb, jac)


def units(text, min_words=4):
    """Sentence / line units with character spans (sentences inside a line, remainder of the line)."""
    out = []
    for lm in re.finditer("[^\\n]+", text):
        line, base, start = (lm.group(), lm.start(), 0)
        for sm in SENT_RE.finditer(line):
            out.append((base + start, base + sm.end(), line[start : sm.end()]))
            start = sm.end()
        if line[start:].strip():
            out.append((base + start, base + len(line), line[start:]))
    return [(a, b, s) for a, b, s in out if len(WORD_RE.findall(s)) >= min_words]


def repeated_spans(text, ref="", min_words=4, jac=0.8):
    """Units of `text` that repeat an earlier unit of ref+text (exactly, or near-duplicate when jac is given)."""
    seen = [norm(s) for _, _, s in units(ref, min_words)] if ref else []
    reps = []
    for a, b, s in units(text, min_words):
        n = norm(s)
        if any((n == p if jac is None else _jac(n, p, jac) for p in seen)):
            reps.append((a, b, s))
        else:
            seen.append(n)
    return reps


def loop_score(text):
    """Share of the text covered by its most frequent n-gram (tokens; char 10-grams for unspaced scripts)."""
    toks = text.split()
    best = 0.0
    if len(toks) >= 12:
        for n in (1, 2, 3, 4, 5, 6):
            if len(toks) < 3 * n:
                break
            g, c = Counter((tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))).most_common(
                1
            )[0]
            if c < 3:
                continue
            if n == 1 and (len(g[0]) < 4 or norm(g[0]) in STOP or (not WORD_RE.search(g[0]))):
                continue
            if n > 1 and (not any((WORD_RE.search(x) for x in g))):
                continue
            best = max(best, c * n / len(toks))
    s = re.sub("\\s+", "", text)
    if len(s) >= 80:
        grams = Counter(
            (
                s[i : i + 10]
                for i in range(len(s) - 9)
                if len(WORD_RE.findall(s[i : i + 10]))
                and sum((ch.isalnum() for ch in s[i : i + 10])) >= 6
            )
        )
        if grams:
            c = grams.most_common(1)[0][1]
            if c >= 3:
                best = max(best, c * 10 / len(s))
    return best


def compression_ratio(text):
    b = text.encode("utf-8", "ignore")
    return len(zlib.compress(b, 9)) / max(1, len(b))


def ascii_share(text):
    return sum((ord(c) < 128 for c in text)) / max(1, len(text))


def token_run(text):
    """A word (>= 2 chars) repeated >= 3 times consecutively ('#grid #grid #grid', 'very very very')."""
    return bool(re.search("(?<!\\S)(\\S{2,})(?:\\s+\\1){2,}(?!\\w)", text))


def is_poemlike(text):
    """Verse: >= 6 lines and at least half of the characters sit in short (<= 70 char) lines, or verse markers."""
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 6:
        return False
    short = sum((len(l) for l in lines if len(l) <= 70)) / max(1, sum((len(l) for l in lines)))
    return short >= 0.5 or bool(re.search("\\b(?:verse|chorus|stanza|refrain)\\b", text, re.I))


def degenerate_reasons(text, ref=""):
    """Repetition / loop diagnostics calibrated on the blind audits: songs and poems may repeat a refrain, prose
    may not repeat a sentence 4+ times or copy >= 35% of itself; near-duplicates count only at >= 50% coverage."""
    reasons = []
    if not text.strip():
        return reasons
    L = max(1, len(text))
    exact = repeated_spans(text, ref, jac=None)
    near = repeated_spans(text, ref, jac=0.8)
    rc_exact = sum((b - a for a, b, _ in exact)) / L
    rc_near = sum((b - a for a, b, _ in near)) / L
    mult = Counter((norm(s) for _, _, s in exact)).most_common(1)[0][1] + 1 if exact else 0
    if is_poemlike(text):
        rep = mult >= 9 or rc_exact >= 0.6
    else:
        rep = mult >= 4 or rc_exact >= 0.3 or rc_near >= 0.35
    if rep:
        reasons.append("repeat")
    if loop_score(text) >= 0.5:
        reasons.append("loop")
    if len(text) >= 300 and ascii_share(text) >= 0.8 and (compression_ratio(text) < 0.14):
        reasons.append("compress")
    if token_run(text):
        reasons.append("run")
    return reasons


def is_degenerate(text, ref=""):
    return bool(degenerate_reasons(text, ref))


def collapse_loops(text):
    """Collapse runs of the same 1..6-token phrase repeated >= 3 times to one occurrence."""
    t = text + " "
    for _ in range(3):
        t2 = re.sub("((?:\\S+\\s+){1,6}?)(?:\\1){2,}", "\\1", t)
        if t2 == t:
            break
        t = t2
    return t.rstrip(" ")


def drop_repeats(text, ref="", min_words=4):
    """Remove units that repeat earlier units (keeping the first occurrence) and tidy whitespace."""
    reps = repeated_spans(text, ref, min_words)
    if not reps:
        return text
    out, pos = ([], 0)
    for a, b, _ in reps:
        out.append(text[pos:a])
        pos = b
    out.append(text[pos:])
    t = "".join(out)
    t = re.sub("[ \\t]{2,}", " ", t)
    t = re.sub(" +\\n", "\n", t)
    t = re.sub("\\n{3,}", "\n\n", t)
    return t.strip()


def clean_piece(piece, ref=""):
    """Generated piece -> loop-collapsed, repeats (also w.r.t. ref) dropped."""
    return drop_repeats(collapse_loops(piece), ref)


def valid_title(inner):
    s = " ".join(inner.strip().strip("<> \"'").split())
    return (
        bool(s)
        and len(ALPHA_RE.findall(s)) >= 1
        and (s.lower() not in TITLE_PLACEHOLDERS)
        and ("<" not in s)
        and (">" not in s)
    )


def title_in(text):
    return any((valid_title(m.group(1)) for m in re.finditer("<<([^\\n<>]*)>>", text)))


def valid_placeholder(inner):
    s = inner.strip()
    if (
        not s
        or len(s) > 60
        or (not ALPHA_RE.search(s))
        or (len("".join(WORD_RE.findall(s))) < 2)
        or PH_BAD.match(s)
    ):
        return False
    if re.search("[\"\\';:{}]", s) or len(s.split()) > 5:
        return False
    return True


def placeholders_in(text):
    return [
        m for m in re.finditer("\\[([^\\[\\]\\n]{0,80})\\]", text) if valid_placeholder(m.group(1))
    ]


def valid_highlight(inner):
    s = inner.strip()
    return len("".join(ALPHA_RE.findall(s))) >= 2 and "highlighted section" not in s.lower()


def highlights_in(text):
    body = re.sub("(?m)^\\s*\\*{3,}\\s*$", "", text)
    body = re.sub("(?m)^\\s*\\* ", "", body)
    return [
        m for m in re.finditer("\\*{1,2}([^*\\n]*)\\*{1,2}", body) if valid_highlight(m.group(1))
    ]


def split_blocks(text, slot, kw=None):
    kw = kw or {}
    if slot == "paragraphs":
        parts = re.split("\\s?\\*\\*\\*\\s?", text)
    elif slot == "sections":
        sp = (kw.get("section_spliter") or "Section").strip()
        parts = re.split("\\s?" + re.escape(sp) + "\\s?\\d+\\s?", text)[1:]
    elif slot == "two":
        parts = text.split("******")
    elif slot == "bullets":
        parts = [
            re.sub("^\\s*[*-]\\s", "", l) for l in text.splitlines() if re.match("^\\s*[*-]\\s", l)
        ]
    elif slot == "nthpara":
        parts = re.split("\\n\\s*\\n", text)
    else:
        return []
    return [x.strip() for x in parts if x.strip()]


def filler_tags(text, iid=None, kw=None, prompt=None, slots=None):
    """Junk-in-format-slot patterns. Generic patterns always; slot-specific ones for the active slots (derived from
    iid, or given explicitly, e.g. from the plan). Calibrated on the blind audits (<= 5% of GENUINE items)."""
    kw = {k: v for k, v in (kw or {}).items() if v is not None}
    slots = set(slots or ()) | ({SLOT_OF[iid]} if iid in SLOT_OF else set())
    tags = set()
    if re.search("\\*{1,2}[^*\\n]*highlighted section[^*\\n]*\\*{1,2}", text, re.I):
        tags.add("junk_highlight")
    for line in text.splitlines():
        if line.strip() and FILLER_LINE.match(line):
            tags.add("filler_block")
    if "\\n\\n" in text:
        tags.add("literal_newline")
    if re.search(
        "\\b(?:Section|SECTION)\\s?\\d+(?:\\s+(?:Section|SECTION)\\s?\\d+)+\\s*$", text.strip()
    ):
        tags.add("trailing_labels")
    if re.search(
        "(?mi)^\\*?\\s*note:\\s*this (?:poem|response|essay|story|text|answer|letter)\\b", text
    ):
        tags.add("meta_note")
    if "placeholders" in slots:
        body = re.sub("```.*?```", "", text, flags=re.S)
        if re.search("\\[\\s*\\]|\\[\\s*(?:\\.\\.\\.|…|x|_+|-+|\\?+)\\s*\\]", body):
            tags.add("empty_placeholder")
        if re.search("\\[[^\\]\\n]*\\]", text) and (not placeholders_in(text)):
            tags.add("no_real_placeholder")
        same = Counter((m.group(1).strip().lower() for m in placeholders_in(text))).most_common(1)
        if same and same[0][1] >= 5:
            tags.add("repeated_placeholder")
    if "highlights" in slots:
        hb = re.sub("(?m)^\\s*\\*{3,}\\s*$", "", text)
        hb = re.sub("(?m)^\\s*\\* ", "", hb)
        hb = re.sub("\\*{2,}", "*", hb)
        if any((not WORD_RE.search(m.group(1)) for m in re.finditer("\\*([^\\n*]*)\\*", hb))):
            tags.add("junk_highlight")
        if re.search("(?m)^\\s*\\* [^*\\n]+\\*\\s*$", text):
            tags.add("junk_highlight")
        for line in hb.splitlines():
            if line.count("*") % 2 == 1 and (not re.fullmatch("\\s*\\*+\\s*", line)):
                tags.add("junk_highlight")
    if "title" in slots:
        ts = re.findall("<<([^\\n<>]*)>>", text)
        if ts and (not any((valid_title(t) for t in ts))):
            tags.add("filler_title")
        if re.search("<<\\s*!--", text):
            tags.add("filler_title")
    if slots & set(BLOCK_SLOTS):
        for para in re.split("\\n\\s*\\n", text):
            s = para.strip()
            if re.fullmatch("\\*+", s) and len(s) not in (3, 6):
                tags.add("lone_symbol_block")
        if re.search("\\*\\* \\* \\*\\*|(?<!\\*)\\*{4,5}(?![*\\w])|(?<![*\\w])\\*{7,}", text):
            tags.add("pseudo_divider")
        for line in text.splitlines():
            s = line.strip()
            if re.fullmatch("\\*+", s) and len(s) not in (3, 6):
                tags.add("junk_divider_line")
        for slot in slots & set(BLOCK_SLOTS):
            bs = split_blocks(text, slot, kw)
            if slot != "bullets" and any((len(WORD_RE.findall(b)) < 3 for b in bs)):
                tags.add("short_block")
            if (
                slot in ("paragraphs", "two", "sections")
                and bs
                and bs[0].rstrip("*").rstrip().endswith(":")
                and (len(WORD_RE.findall(bs[0])) <= 15)
            ):
                tags.add("intro_block")
            for i in range(len(bs)):
                if any((near_dup(bs[i], bs[j]) for j in range(i))):
                    tags.add("duplicate_block")
            if prompt:
                pn = norm(prompt)
                if any((len(WORD_RE.findall(b)) >= 6 and norm(b) in pn for b in bs)):
                    tags.add("prompt_copy")
    if "nthpara" in slots and kw.get("first_word"):
        if re.search("(?m)^\\s*" + re.escape(kw["first_word"]) + "\\s+\\d+\\s*[:.)]", text, re.I):
            tags.add("label_glue")
    if "repeat_prompt" in slots or kw.get("prompt_to_repeat"):
        p = norm(kw.get("prompt_to_repeat") or (prompt or ""))
        if p and echo_only(text, p):
            tags.add("echo_only")
    elif prompt:
        pn, n = (norm(prompt), norm(text))
        if pn and (n.count(pn) >= 2 or echo_only(text, pn) or prompt_echo(text, prompt)):
            tags.add("echo_only")
    if "postscript" in slots:
        m = re.search("p\\.\\s?p?\\.?\\s?s\\.?(.*)", text, re.I | re.S)
        if m:
            after = m.group(1)
            if len(WORD_RE.findall(after)) < 2:
                tags.add("filler_postscript")
            if len(WORD_RE.findall(after)) > 100:
                tags.add("ps_midtext")
            if repeated_spans(after, text[: m.start()], jac=None):
                tags.add("filler_postscript")
    return tags


def prompt_echo(text, prompt, max_rest=40):
    """Echo of an embedded request: >= 8 words of the response are sentences copied from the prompt and either
    (almost) nothing else follows, or the copied request appears twice and fewer than max_rest other words exist."""
    pn, n = (norm(prompt), norm(text))
    total = len(WORD_RE.findall(text))
    if not pn or not total:
        return False
    cop = [norm(s) for _, _, s in units(text) if norm(s) in pn]
    copied = sum((len(c.split()) for c in cop))
    rest = total - copied
    if copied < 8:
        return False
    twice = any((len(c.split()) >= 6 and n.count(c) >= 2 for c in cop))
    return rest < 2 or (twice and rest < max_rest)


def echo_only(text, prompt_norm, min_words=2):
    """The answer is the prompt repeated (once or more) with (almost) nothing else."""
    n = norm(text)
    if not prompt_norm:
        return False
    rest = n.replace(prompt_norm, " ")
    return len(rest.split()) < min_words


def splice_tags(text, new_words=None, keywords=(), forbidden=()):
    """Word-level splice patterns. With new_words (a Counter of words introduced by certification) only those
    words are checked; otherwise every word of the text."""
    tags = set()
    words = list(new_words) if new_words is not None else WORD_RE.findall(text)
    for w in words:
        lw = w.lower()
        if re.search("([^\\W\\d_])\\1\\1", w) and (not TRIPLE_OK.match(w)):
            tags.add("triple_letter")
        if CYR_GREEK.search(w) and LATIN.search(w):
            tags.add("mixed_script")
        if ALPHA_RE.search(w) and len({_script(c) for c in w if c.isalpha()} - {None}) > 1:
            tags.add("mixed_script")
        for k in keywords:
            if k and k in lw and (lw != k) and (not _inflection(k, lw)):
                tags.add("glued_keyword")
        for f in forbidden:
            if f and f in lw and (lw != f):
                tags.add("glued_forbidden")
    return tags


def _script(c):
    o = ord(c)
    if o < 592:
        return "latin"
    if 880 <= o < 1024:
        return "greek"
    if 1024 <= o < 1328:
        return "cyrillic"
    if 2304 <= o < 2432:
        return "devanagari"
    if 2688 <= o < 2816:
        return "gujarati"
    if 3200 <= o < 3328:
        return "kannada"
    if 3584 <= o < 3712:
        return "thai"
    if 44032 <= o < 55216 or 4352 <= o < 4608:
        return "hangul"
    if 19968 <= o < 40960 or 12352 <= o < 12544:
        return "cjk"
    if 1536 <= o < 1792:
        return "arabic"
    return None


def _inflection(k, w):
    sufs = (
        "s",
        "es",
        "ed",
        "d",
        "ing",
        "ly",
        "er",
        "ers",
        "est",
        "ness",
        "ment",
        "ments",
        "ful",
        "less",
        "ion",
        "ions",
    )
    if any((w == k + s for s in sufs)):
        return True
    if k.endswith("e") and any(
        (w == k[:-1] + s for s in ("ing", "ed", "er", "ers", "ation", "ations"))
    ):
        return True
    if k.endswith("y") and any((w == k[:-1] + s for s in ("ies", "ied", "ier", "iest", "ily"))):
        return True
    return False


def detect(text, iid=None, kw=None, prompt=None, keywords=(), forbidden=(), slots=None):
    """All program-side loophole detectors on a finished response: repetition + filler slots + splices
    (keyword gluing is only checked when keywords/forbidden are given; the certification guard checks them on
    the words a MAP edit introduced)."""
    tags = (
        set(degenerate_reasons(text))
        | filler_tags(text, iid, kw, prompt, slots)
        | splice_tags(text, None, keywords, forbidden)
    )
    return sorted(tags)


NLTK_CORPORA = Path(os.environ.get("NLTK_DATA", str(PROJECT_ROOT / "assets/nltk"))) / "corpora"
EXTRA_WORDS = frozenset(
    "email emails online website websites webpage app apps smartphone smartphones blog blogs blogger bloggers internet startup startups wifi gmail iphone ipad okay ok selfie selfies hashtag hashtags podcast podcasts software hardware login logout username usernames password passwords chatbot chatbots ai covid vlog vlogs youtube google facebook instagram tiktok twitter linkedin emoji emojis laptop laptops smartwatch skincare workflow workflows dataset datasets cybersecurity ecommerce fintech blockchain crypto online offline livestream streaming esports gameplay teammate teammates healthcare lifestyle lifestyles mindset feedback".split()
)
_LEX = None


def lexicon():
    """English word set: NLTK 'words' (Webster 2nd, 236k) + CMUdict (134k, inflected forms) + a few modern words.
    Empty when the corpora are missing -- every lexicon-based check is then disabled (never a false positive)."""
    global _LEX
    if _LEX is None:
        s = set()
        try:
            for name in ("words/en", "words/en-basic"):
                with open(NLTK_CORPORA / name, encoding="utf-8", errors="ignore") as f:
                    s.update((w.strip().lower() for w in f if w.strip()))
            with open(NLTK_CORPORA / "cmudict/cmudict", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    w = line.split(" ", 1)[0].lower()
                    if w.replace("'", "").isalpha():
                        s.add(w)
            s |= EXTRA_WORDS
        except OSError:
            s = set()
        _LEX = frozenset(s) if len(s) > 100000 else frozenset()
    return _LEX


SUFFIX_RULES = (
    ("ies", "y"),
    ("ied", "y"),
    ("ier", "y"),
    ("iers", "y"),
    ("iest", "y"),
    ("ily", "y"),
    ("iness", "y"),
    ("es", ""),
    ("s", ""),
    ("ed", ""),
    ("ed", "e"),
    ("ing", ""),
    ("ing", "e"),
    ("ly", ""),
    ("er", ""),
    ("er", "e"),
    ("ers", ""),
    ("ers", "e"),
    ("est", ""),
    ("est", "e"),
    ("ness", ""),
    ("ment", ""),
    ("ments", ""),
    ("ful", ""),
    ("less", ""),
    ("able", ""),
    ("able", "e"),
    ("ably", "e"),
    ("ation", "e"),
    ("ations", "e"),
    ("ation", ""),
    ("ization", "ize"),
    ("ize", ""),
    ("izes", ""),
    ("ized", ""),
    ("izing", ""),
    ("ity", ""),
    ("ism", ""),
    ("isms", ""),
    ("ist", ""),
    ("ists", ""),
    ("al", ""),
    ("ally", ""),
    ("ically", "ic"),
    ("ive", ""),
    ("ives", ""),
    ("ship", ""),
    ("ships", ""),
    ("hood", ""),
    ("wise", ""),
    ("like", ""),
    ("ish", ""),
)
WORD_PREFIXES = (
    "un",
    "re",
    "non",
    "pre",
    "over",
    "under",
    "anti",
    "co",
    "de",
    "dis",
    "mis",
    "multi",
    "inter",
    "sub",
    "super",
    "self",
    "semi",
    "post",
    "mid",
    "out",
    "up",
    "eco",
    "micro",
    "mega",
    "cyber",
    "bio",
    "tele",
)


def _in_lex(lw, L, depth=0):
    if lw in L:
        return True
    for suf, rep in SUFFIX_RULES:
        if lw.endswith(suf) and len(lw) - len(suf) >= 3:
            b = lw[: -len(suf)] + rep
            if b in L:
                return True
            if len(b) >= 4 and b[-1] == b[-2] and (b[:-1] in L):
                return True
    if depth == 0:
        for pre in WORD_PREFIXES:
            if lw.startswith(pre) and len(lw) - len(pre) >= 4 and _in_lex(lw[len(pre) :], L, 1):
                return True
        for i in range(4, len(lw) - 3):
            if lw[:i] in L and _in_lex(lw[i:], L, 1):
                return True
    return False


_WORD_CACHE = {}


def is_word(w):
    """True for English lexicon words (with inflection / prefix / compound rules), for tokens <= 2 letters,
    non-alphabetic tokens and non-ASCII tokens (undecidable -> never flagged), and when no lexicon is available."""
    lw = w.lower()
    if len(lw) <= 2 or not lw.isascii() or (not lw.isalpha()):
        return True
    L = lexicon()
    if not L:
        return True
    r = _WORD_CACHE.get(lw)
    if r is None:
        r = _WORD_CACHE[lw] = _in_lex(lw, L)
    return r


def ascii_letter_share(text):
    letters = [c for c in text if c.isalpha()]
    return sum((c.isascii() for c in letters)) / max(1, len(letters))


_STOP_EN = None


def stop_en():
    global _STOP_EN
    if _STOP_EN is None:
        try:
            with open(NLTK_CORPORA / "stopwords/english", encoding="utf-8") as f:
                _STOP_EN = frozenset((w.strip().lower() for w in f if w.strip()))
        except OSError:
            _STOP_EN = frozenset(STOP)
    return _STOP_EN


def is_english(text, plan_=None):
    """English prose: no other target language, >= 90% ASCII letters and (for >= 15 tokens) >= 12% English stopwords
    (French / Spanish / German written in ASCII is not English; lexicon checks are skipped for it)."""
    lang = (plan_ or {}).get("language")
    if lang not in (None, "en") or ascii_letter_share(text) < 0.9:
        return False
    toks = [t.lower() for t in ALPHA_RE.findall(text)]
    if len(toks) < 15:
        return True
    return sum((t in stop_en() for t in toks)) / len(toks) >= 0.12


def real_words(s, english=True):
    """Words that carry content: alphabetic tokens that are lexicon words, capitalised names or 'a'/'I' (English);
    any letter-bearing token otherwise."""
    toks = ALPHA_RE.findall(s)
    if not english:
        return len(toks)
    return sum(
        (1 for t in toks if len(t) >= 2 and (is_word(t) or t[:1].isupper()) or t in ("a", "A", "I"))
    )


def prose_only(text):
    """Text without fenced / inline code, URLs, e-mail addresses, HTML tags and code-like lines."""
    t = re.sub("```.*?(?:```|$)", " ", text, flags=re.S)
    t = re.sub("`[^`\\n]*`", " ", t)
    t = re.sub("(?:https?://|www\\.)\\S+|\\S+@\\S+|</?[A-Za-z][^>\\n]*>", " ", t)
    return "\n".join(
        (
            l
            for l in t.splitlines()
            if not re.search(
                "[{};=]|\\w\\(|\\)\\s*$|^\\s*(?:import|def|class|public|private|return|#include)\\b",
                l,
            )
        )
    )


def nonword_tokens(text):
    """Lower-case (or all-caps text) ASCII alphabetic tokens of >= 3 letters that are not English words (prose only;
    identifiers such as 'javax.servlet' or 'snake_case' are skipped)."""
    text = prose_only(text)
    toks = re.findall("(?<![\\w'./@#\\\\-])[A-Za-z]{3,}(?![\\w./@\\\\-])", text)
    letters = [c for c in text if c.isalpha() and c.isascii()]
    caps_text = letters and sum((c.isupper() for c in letters)) / len(letters) > 0.7
    cand = [t for t in toks if caps_text or t.islower()]
    return (cand, [t for t in cand if not is_word(t)])


LIST_ITEM = re.compile("^\\s*(?:\\d{1,3}[.)]|[*\\-•+]|[a-hA-H][.)])\\s+\\S")
BARE_MARK = re.compile("^\\s*(?:\\d{1,3}[.)]|[*\\-•+]|[a-hA-H][.)]|#{1,6})\\s*$")
LABEL_ITEM = re.compile(
    "^\\s*(?:\\d{1,3}[.)]|[*\\-•+]|#{1,6})?\\s*[^\\s:.!?]+(?:[ \\t]+[^\\s:.!?]+){0,4}[ \\t]*:\\s*$"
)
CONCL = re.compile(
    "(?i)(?:^|(?<=[.!?]\\s)|(?<=\\n))\\W{0,3}(?:in conclusion|to conclude|in summary|to sum up|in closing|to summarize|all in all)\\b"
)
META_RE = re.compile(
    "(?i)\\b(?:your task|the task is|the request|the prompt|the instructions?|repeat(?:ed|ing)? the (?:request|prompt|text)|word for word|my (?:response|answer) (?:will|should|must|is)|this response|the (?:last|final|first) sentence|i will (?:evaluate|now|repeat|write|answer)|part #?\\d+\\s*:|(?:the|this) (?:word|keyword|phrase|letter) [\\\"'])"
)
META_KW = re.compile(
    "(?i)\\b(?:the (?:word|words|keyword|keywords|term|terms|phrase)|keywords?|use[sd]? the word|include[sd]? the words?|mention(?:s|ed)? the word)\\b"
)
META_HL = re.compile(
    "(?i)^\\W*(?:(?:sarcasm|important|key|main|this is an?|the|my|a)\\s+)?(?:highlight(?:ed|s|ing)?|light|section|part|note|point|heading|header|text|emphasis|bold|italic)(?:\\s+(?:section|part|point|text|here|one|two|three))*\\s*(?:#|no\\.?)?\\s*\\d*\\W*$"
)
SYMBOL_GLUE = re.compile("[^\\W\\d_][=_~#*]{6,}|[=_~#*]{6,}[^\\W\\d_]")
CTRL_JUNK = re.compile("[\\r\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f](?=\\S)")


def _nonempty_lines(text):
    return [l for l in text.splitlines() if l.strip()]


def skeleton_lines(text):
    """(bare list markers, label-only items that introduce no content). A label-only item is empty when the next
    non-empty line is another label-only / bare item or the text ends."""
    lines = _nonempty_lines(text)
    bare = [
        i for i, l in enumerate(lines) if BARE_MARK.match(l) and l.strip() not in ("***", "******")
    ]
    lab = [i for i, l in enumerate(lines) if LABEL_ITEM.match(l) and (not BARE_MARK.match(l))]
    labset, bset = (set(lab), set(bare))
    empty_lab = [i for i in lab if i + 1 >= len(lines) or i + 1 in labset or i + 1 in bset]
    return (bare, empty_lab, lines)


def strip_skeleton(text):
    """Remove bare list-marker lines and label-only items without content (honest repair, never adds anything)."""
    bare, empty_lab, lines = skeleton_lines(text)
    drop = {lines[i] for i in bare} if len(bare) >= 1 else set()
    if len(empty_lab) >= 3:
        drop |= {lines[i] for i in empty_lab}
    if not drop:
        return text
    out = [l for l in text.splitlines() if not (l.strip() and l in drop)]
    t = "\n".join(out)
    t = re.sub("(?:[ \\t]*\\r?\\n){3,}", "\n\n", t)
    return t.strip("\n")


def official_blocks(text, kind, kw=None):
    """Blocks exactly as the official checker splits them (marker / header removed)."""
    kw = kw or {}
    if kind == "paragraphs":
        return [b for b in re.split("\\s?\\*\\*\\*\\s?", text)]
    if kind == "two":
        return text.split("******")
    if kind == "sections":
        sp = (kw.get("section_spliter") or "Section").strip()
        return re.split("\\s?" + re.escape(sp) + "\\s?\\d+\\s?", text)[1:]
    if kind == "bullets":
        return [
            re.sub("^\\s*(?:\\*|-)\\s?", "", l)
            for l in text.splitlines()
            if re.match("^\\s*\\*[^\\*]", l) or re.match("^\\s*-", l)
        ]
    if kind == "nthpara":
        return re.split("\\n\\s*\\n", text)
    return []


def _grams(s, n=4):
    w = norm(s).split()
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def recycled_units(text, start=0, min_words=5, contain=0.6, jac=0.6):
    """Sentence units (after offset `start`) that mostly recycle earlier text: >= `contain` of their word 4-grams
    already occur earlier, or word-set Jaccard >= `jac` with an earlier unit."""
    us = units(text, min_words)
    seen_g, seen_n, out = (set(), [], [])
    for a, b, s in us:
        g = _grams(s)
        n = norm(s)
        if a >= start:
            rec = (
                g and len(g & seen_g) / len(g) >= contain or any((_jac(n, p, jac) for p in seen_n))
            )
            if rec:
                out.append((a, b, s))
        seen_g |= g
        seen_n.append(n)
    return out


def genuine_sentence_count(text):
    """Sentences with >= 3 alphabetic words; bare list numbers / markers and skeleton lines never count."""
    t = re.sub("(?m)^\\s*(?:\\d{1,3}[.)]|[a-hA-H][.)])\\s*", "", text)
    n = 0
    for _, _, s in units(t, 0):
        if len(ALPHA_RE.findall(s)) >= 3:
            n += 1
    return n


def loose_sentence_count(text):
    """Upper bound in the spirit of the official splitter: every '.', '!' or '?' run counts (list numbers too)."""
    return len(re.findall("[.!?]+", text))


def hl_official(text):
    out = []
    for m in re.finditer("\\*[^\\n\\*]*\\*", text):
        if m.group(0).strip("*").strip():
            out.append((m.start(), m.end(), m.group(0).strip("*").strip()))
    for m in re.finditer("\\*\\*[^\\n\\*]*\\*\\*", text):
        if m.group(0)[2:-2].strip():
            out.append((m.start(), m.end(), m.group(0)[2:-2].strip()))
    return out


def hl_valid4(text, a, b, inner, english=True):
    if META_HL.match(inner) or "highlighted section" in inner.lower():
        return False
    if a > 0 and text[a - 1].isalnum():
        return False
    if b < len(text) and text[b].islower() and inner[-1:].isalpha():
        return False
    toks = ALPHA_RE.findall(inner)
    if not toks:
        return False
    if english:
        return any((len(t) >= 3 and (is_word(t) or t[:1].isupper()) for t in toks))
    return any((len(t) >= 2 for t in toks))


def ph_official(text):
    return [(m.start(), m.end(), m.group(1)) for m in re.finditer("\\[(.*?)\\]", text)]


def ph_valid4(text, a, b, inner, english=True):
    if not valid_placeholder(inner):
        return False
    if a > 0 and (text[a - 1].isalnum() or text[a - 1] == "'"):
        return False
    if b < len(text) and text[b].isalnum():
        return False
    s = inner.strip()
    if not (s[:1].isalnum() or s[:1] in "$#@&(+"):
        return False
    toks = ALPHA_RE.findall(s)
    if len(toks) == 1 and len(toks[0]) <= 2 and toks[0].islower():
        return False
    if english and any((t.islower() and len(t) >= 3 and (not is_word(t)) for t in toks)):
        return False
    return True


def title_spans(text):
    return [(m.start(), m.end(), m.group(1)) for m in re.finditer("<<([^\\n]+?)>>", text)]


def title_valid4(text, a, b, inner, english=True):
    """A title: valid_title content of <= 12 words that does not end like a clause (',', ';', ' and'), starts with a
    capital / digit / quote unless the whole text is lower-case, and is not a fragment cut out of a running sentence
    (lower-case word right before '<<' and right after '>>')."""
    if not valid_title(inner):
        return False
    s = " ".join(inner.split())
    if (
        len(s.split()) > 12
        or s.endswith((",", ";"))
        or re.search("(?i)\\b(?:and|or|the|a|of|to|with)$", s)
    ):
        return False
    letters = [c for c in text if c.isalpha()]
    lower_text = letters and sum((c.islower() for c in letters)) / len(letters) > 0.97
    if not lower_text and (
        not (s[:1].isupper() or s[:1].isdigit() or s[:1] in "\"'“*#" or (not s[:1].isascii()))
    ):
        return False
    if english:
        toks = ALPHA_RE.findall(s)
        if toks and sum((1 for t in toks if is_word(t) or t[:1].isupper())) / len(toks) < 0.5:
            return False
    ls = text.rfind("\n", 0, a) + 1
    le = text.find("\n", b)
    le = len(text) if le < 0 else le
    before, after = (text[ls:a], text[b:le])
    if re.search("[a-z]\\w*[ \\t]*$", before) and re.match("[ \\t]*[a-z]", after):
        return False
    return True


def kw_occurrences(text, kw):
    pat = re.compile(
        "(?<![A-Za-z])" + re.escape(kw) + "(?:s|es|ed|d|ing|er|ers|ly|'s)?(?![A-Za-z])", re.I
    )
    return list(pat.finditer(text))


def genuine_kw_count(text, kw, start=0, english=True):
    """Keyword uses counted like the official checker (case-insensitive substring) but only when genuine: after
    `start` (not in the echoed prompt); inside a real word (the keyword itself or a longer English word containing
    it, e.g. 'overdose' for 'dose', not a glued non-word like 'ISNEAKER'); not glued to symbol runs
    ('disappointed====='); in a unit with >= 2 other words; not in a meta-statement ('the word "dose"'); not one
    half of a split word ('overd dose')."""
    reasons = Counter()
    n = 0
    spans = [(a, b, s) for a, b, s in units(text, 0)]
    for m in re.finditer(re.escape(kw), text, re.I):
        if m.start() < start:
            reasons["echo"] += 1
            continue
        ts, te = (m.start(), m.end())
        while ts > 0 and text[ts - 1].isalpha():
            ts -= 1
        while te < len(text) and text[te].isalpha():
            te += 1
        tok = text[ts:te]
        if english and " " not in kw and (tok.lower() != kw) and (not is_word(tok)):
            reasons["glued"] += 1
            continue
        unit = next((s for a, b, s in spans if a <= m.start() < b), "")
        after, before = (text[te : te + 3], text[max(0, ts - 3) : ts])
        if re.match("[=_~#\\-]{3}|\\*{3}", after) or re.search("(?:[=_~#\\-]{3}|\\*{3})$", before):
            reasons["glued"] += 1
            continue
        if len(ALPHA_RE.findall(unit)) - len(ALPHA_RE.findall(kw)) < 2 and (
            not re.match("^\\s*(?:[*\\-]\\s+)?(?:\\*\\*|#{1,6}\\s|__)", unit)
        ):
            reasons["no_sentence"] += 1
            continue
        if META_KW.search(unit) or re.search("[\"“‘\\']" + re.escape(kw) + "[\"”’\\']", unit, re.I):
            reasons["meta"] += 1
            continue
        if english and tok.lower() == kw:
            pm = re.search("([A-Za-z]+) ?$", text[max(0, ts - 30) : ts])
            nm = re.match(" ?([A-Za-z]+)", text[te : te + 30])
            w = tok.lower()
            if (
                pm
                and (not is_word(pm.group(1)))
                and (len(pm.group(1)) >= 2)
                and any(
                    (
                        len(c) >= 5 and is_word(c)
                        for c in (pm.group(1) + w, pm.group(1)[:-1] + w, pm.group(1) + w[1:])
                    )
                )
            ):
                reasons["split"] += 1
                continue
            if (
                nm
                and (not is_word(nm.group(1)))
                and (len(nm.group(1)) >= 2)
                and any(
                    (
                        len(c) >= 5 and is_word(c)
                        for c in (w + nm.group(1), w[:-1] + nm.group(1), w + nm.group(1)[1:])
                    )
                )
            ):
                reasons["split"] += 1
                continue
        n += 1
    return (n, reasons)


def letter_mangled(text, letter, relation):
    """Non-words that become English words when the constrained letter is put back ('cheserfield', 'terformance',
    'stri') -- relation 'less than' -- or taken out ('tthe') -- relation 'at least'."""
    L = lexicon()
    letter = letter.lower()
    if not L:
        return []
    out = []
    for w in set(re.findall("(?<![A-Za-z])[A-Za-z]{3,}(?![A-Za-z])", text)):
        lw = w.lower()
        if is_word(lw):
            continue
        if relation == "less than":
            if letter in lw:
                continue
            cands = {lw[:i] + letter + lw[i:] for i in range(len(lw) + 1)} | {
                lw[:i] + letter + lw[i + 1 :] for i in range(len(lw))
            }
        else:
            if letter not in lw:
                continue
            idx = [i for i, c in enumerate(lw) if c == letter]
            cands = {lw[:i] + lw[i + 1 :] for i in idx} | {
                lw[:i] + c + lw[i + 1 :]
                for i in idx
                for c in "abcdefghijklmnopqrstuvwxyz"
                if c != letter
            }
        if any((len(c) >= 4 and _in_lex(c, L) for c in cands)):
            out.append(w)
    return sorted(out)


def script_soup(text):
    """Pseudo-script heuristics for non-Latin answers (whitespace-delimited words, so Indic vowel signs stay inside
    their word): word-initial combining marks, doubled base letters inside a word (Indic / Arabic scripts), and stem
    recycling (one 3-character word start covering >= 35% of the long words). Calibrated on the audit items: every
    GENUINE non-Latin answer has stem share <= 0.19 and no doubled letters."""
    import unicodedata

    words = [
        w
        for w in re.findall(
            "[^\\s\\d.,!?;:()\\[\\]{}\"\\'\\u00ab\\u00bb\\u060c\\u0964\\u0965\\u201c\\u201d\\u2018\\u2019\\u2014\\u2013*#_/-]+",
            text,
        )
        if any((ord(c) >= 880 and (not 7680 <= ord(c) < 7936) for c in w))
    ]
    if len(words) < 15:
        return None
    bad = sum((1 for w in words if unicodedata.category(w[0]).startswith("M")))
    dbl = sum(
        (
            1
            for w in words
            if any(
                (
                    w[i] == w[i + 1]
                    and 1536 <= ord(w[i]) < 3584
                    and (unicodedata.category(w[i]) == "Lo")
                    for i in range(len(w) - 1)
                )
            )
        )
    )
    long_w = [w for w in words if len(w) >= 4]
    top = Counter((w[:3] for w in long_w)).most_common(1)
    rec = top[0][1] / len(long_w) if top and len(long_w) >= 10 else 0.0
    if bad / len(words) >= 0.05 or dbl / len(words) >= 0.15 or rec >= 0.35:
        return dict(initial_marks=bad, doubled=dbl, stem_share=round(rec, 2), words=len(words))
    return None


VI_TONES = {"̀": "huyen", "́": "sac", "̃": "nga", "̉": "hoi", "̣": "nang"}
VI_INITIALS = sorted(
    [
        "",
        "b",
        "c",
        "ch",
        "d",
        "đ",
        "g",
        "gh",
        "gi",
        "h",
        "k",
        "kh",
        "l",
        "m",
        "n",
        "ng",
        "ngh",
        "nh",
        "p",
        "ph",
        "qu",
        "r",
        "s",
        "t",
        "th",
        "tr",
        "v",
        "x",
    ],
    key=len,
    reverse=True,
)
VI_NUCLEI = {
    "a",
    "ă",
    "â",
    "e",
    "ê",
    "i",
    "y",
    "o",
    "ô",
    "ơ",
    "u",
    "ư",
    "ia",
    "iê",
    "yê",
    "ua",
    "uô",
    "ưa",
    "ươ",
    "oa",
    "oă",
    "oe",
    "uê",
    "uy",
    "uâ",
    "uyê",
    "uya",
    "oo",
    "uơ",
}
VI_FINALS = ("ng", "nh", "ch", "c", "m", "n", "p", "t", "i", "y", "o", "u", "")
VI_NEED_FINAL = {"ă", "â", "iê", "yê", "uô", "ươ", "uyê", "oă", "uâ"}
VI_NO_FINAL = {"ia", "ua", "ưa", "uya"}
VI_GLIDE_OK = {
    ("a", "i"),
    ("a", "y"),
    ("â", "y"),
    ("â", "u"),
    ("a", "o"),
    ("a", "u"),
    ("e", "o"),
    ("ê", "u"),
    ("i", "u"),
    ("o", "i"),
    ("ô", "i"),
    ("ơ", "i"),
    ("u", "i"),
    ("ư", "i"),
    ("ư", "u"),
    ("iê", "u"),
    ("yê", "u"),
    ("uô", "i"),
    ("ươ", "i"),
    ("ươ", "u"),
    ("oa", "i"),
    ("oa", "y"),
    ("oa", "o"),
    ("oe", "o"),
    ("uâ", "y"),
    ("uy", "u"),
}


def vi_syllable_ok(w):
    """Vietnamese orthographic syllable check: initial + nucleus + final from the standard inventories, one tone
    mark, stop finals (c ch p t) only with the sac / nang tones, nuclei that need / forbid a final, valid glides."""
    d = unicodedata.normalize("NFD", w.lower())
    tones = [c for c in d if c in VI_TONES]
    if len(tones) > 1:
        return False
    base = unicodedata.normalize("NFC", "".join((c for c in d if c not in VI_TONES)))
    for ini in VI_INITIALS:
        if not base.startswith(ini):
            continue
        rest = base[len(ini) :]
        if ini == "gi" and (not rest or rest[0] not in "aăâeêoôơuư"):
            rest = "i" + rest
        if ini == "qu" and rest[:1] not in ("a", "ă", "â", "e", "ê", "i", "y", "o", "ơ"):
            continue
        for fin in VI_FINALS:
            if not rest.endswith(fin):
                continue
            nuc = rest[: len(rest) - len(fin)] if fin else rest
            if nuc not in VI_NUCLEI:
                continue
            if nuc in VI_NEED_FINAL and (not fin):
                continue
            if nuc in VI_NO_FINAL and fin:
                continue
            if fin in ("i", "y", "o", "u") and (nuc, fin) not in VI_GLIDE_OK:
                continue
            if fin in ("ch", "nh") and nuc not in ("a", "ê", "i", "oa", "uy", "e", "y"):
                continue
            if (
                fin in ("c", "ch", "p", "t")
                and tones
                and (VI_TONES[tones[0]] not in ("sac", "nang"))
            ):
                continue
            return True
    return False


def vi_pseudo(text):
    """Vietnamese pseudo-syllables: >= 5% of the diacritic-bearing Latin syllables (>= 20 of them) and >= 3 distinct
    ones are orthographically impossible. Genuine Vietnamese audit answers have 0 invalid syllables."""
    toks = [
        t
        for t in re.findall("[^\\W\\d_]+", text)
        if not t.isascii() and all((ord(c) < 592 or 7680 <= ord(c) < 7936 for c in t))
    ]
    if len(toks) < 20:
        return None
    if sum((1 for t in toks if re.search("[ơưđƠƯĐẠ-ỹ]", t))) < 0.5 * len(toks):
        return None
    bad = [t for t in toks if not vi_syllable_ok(t)]
    if len(bad) >= 0.05 * len(toks) and len(set(bad)) >= 3:
        return dict(invalid=len(bad), syllables=len(toks))
    return None


SEP_END_OK = re.compile("[.!?:;\"\\'”)\\]>][*_]*\\s*$")


def separator_issues(tail):
    """'***' paragraph separators: a separator glued to text must follow a sentence end (or a blank line when it
    starts a line); a separator between two list items splits a list."""
    tags = {}
    lines = tail.split("\n")
    pos = 0
    for i, l in enumerate(lines):
        for m in re.finditer("(?<!\\*)\\*\\*\\*(?!\\*)", l):
            before = l[: m.start()]
            prv = next((x for x in reversed(lines[:i]) if x.strip()), "")
            blank_before = i == 0 or not lines[i - 1].strip()
            if before.strip() and (not re.fullmatch("\\s*[\\[(<\"\\']+\\s*", before)):
                if not SEP_END_OK.search(before):
                    tags["B:inline_sep"] = l.strip()[:20]
            elif l.strip() != "***" and (not blank_before) and prv and (not SEP_END_OK.search(prv)):
                tags["B:inline_sep"] = l.strip()[:20]
            nxt_txt = l[m.end() :].strip() or next((x for x in lines[i + 1 :] if x.strip()), "")
            nxt_line = next((x for x in lines[i + 1 :] if x.strip()), "")
            if (
                prv
                and LIST_ITEM.match(prv)
                and (
                    LIST_ITEM.match(nxt_txt)
                    or (nxt_line and LIST_ITEM.match(nxt_line) and l[m.end() :].strip())
                )
            ):
                tags["B:sep_mid_list"] = 1
            if (
                l.strip() == "***"
                and prv
                and re.search(
                    "(?:[,;]|\\b(?:and|or|but|the|a|an|of|to|with|for|that|which|who))\\s*$",
                    prv.strip(),
                    re.I,
                )
            ):
                tags["B:sep_mid_sentence"] = 1
    return tags


def markup_profile(text, plan_=None):
    """(stray-markup lines, mid-line asterisks) -- used as a certification diff: a MAP edit that leaves a line with
    unbalanced markup ('**Attention to detail.', 'SECTION 1]', '<<fragment') or inserts '*' inside a line when no
    highlight is requested ('Aaron (* Ethan Eth* Grayson') is junk structure."""
    stray = mid = 0
    for l in text.splitlines():
        s = l.strip()
        if not s or s in ("***", "******") or re.fullmatch("[*\\-_=#]{3,}", s):
            continue
        body = re.sub("^(?:[*\\-\\u2022+]|\\d{1,3}[.)])\\s+", "", s)
        if (
            body.count("*") % 2
            or body.count("**") % 2
            or body.count("[") != body.count("]")
            or (body.count("<<") != body.count(">>"))
        ):
            stray += 1
        mid += body.count("*")
    if (plan_ or {}).get("highlights"):
        mid = 0
    return (stray, mid)


def g4_issues(text, plan_, prefix=None):
    """All comparative guard tags for a finished (or partial) response under its plan."""
    plan_ = plan_ or {}
    tags = {}
    prefix = plan_.get("prefix", "") if prefix is None else prefix
    start = len(prefix) if prefix and text.startswith(prefix) else 0
    tail = text[start:]
    if not tail.strip():
        if prefix:
            tags["H:echo_soup"] = "no answer after the echo"
        return tags
    eng = is_english(tail, plan_)
    b = plan_.get("blocks") or {}
    kind = b.get("kind")
    for r in degenerate_reasons(tail):
        tags["I:" + r] = 1
    if re.search("(?<![\\w])([A-Za-z])(?:[ \\t]+\\1){2,}(?![\\w])", tail):
        tags["I:letter_run"] = 1
    if SYMBOL_GLUE.search(tail):
        tags["I:symbol_run"] = 1
    if eng:
        cand, bad = nonword_tokens(tail)
        if len(cand) >= 20 and len(bad) / len(cand) >= 0.2:
            tags["I:pseudo_words"] = f"{len(bad)}/{len(cand)}"
    else:
        ss = script_soup(tail) or (
            vi_pseudo(tail) if plan_.get("language") in (None, "vi") else None
        )
        if ss:
            tags["I:pseudo_script"] = ss
    bare, empty_lab, _ = skeleton_lines(tail)
    if len(bare) >= 2 or len(empty_lab) >= 3:
        tags["A:skeleton"] = f"{len(bare)} bare, {len(empty_lab)} empty labels"
    last = (_nonempty_lines(tail) or [""])[-1]
    if BARE_MARK.match(last) and last.strip() not in ("***", "******"):
        tags["A:fragment_end"] = "bare marker at the end"
    elif (
        re.search("(?<![\\w])([A-Za-z])(?:[ \\t]+[A-Za-z]){1,}[ \\t]*$", last)
        and len(ALPHA_RE.findall(last)) > 2
        and (not re.search("(?:\\b[A-Z]\\.[ \\t]*){2,}$", last.strip()))
    ):
        tags["A:fragment_end"] = "single-letter tail"
    if kind:
        kwb = (
            dict(section_spliter=b.get("block_prefix", "Section {i}").split(" {i}")[0].strip())
            if kind == "sections"
            else {}
        )
        bs = [x.strip() for x in official_blocks(tail, kind, kwb)]
        bs = [x for x in bs if x]
        mw = 1 if kind == "bullets" else 5
        if any((real_words(strip_skeleton(x), eng) < mw for x in bs)):
            tags["A:empty_block"] = "block without content"
        for i in range(len(bs)):
            for j in range(i):
                gi, gj = (_grams(bs[i]), _grams(bs[j]))
                if near_dup(bs[i], bs[j]) or (len(gi) >= 8 and len(gi & gj) / len(gi) >= 0.7):
                    tags["A:dup_block"] = f"{j}~{i}"
    if kind == "paragraphs":
        tags.update(separator_issues(tail))
    if kind == "bullets":
        bl = [x.strip() for x in official_blocks(tail, "bullets")]
        if any(
            (
                re.search("\\w[ \\t]+\\d{1,2}[.)]?[ \\t]+[A-Z][a-z]", x)
                or re.match("\\d{1,2}[ \\t]+[A-Z][a-z]", x)
                for x in bl
            )
        ):
            tags["B:bullet_merged"] = "items squashed into one bullet"
        wl = [len(WORD_RE.findall(x)) for x in bl]
        if (
            len(wl) >= 2
            and max(wl) >= 60
            and (sorted(wl)[-2] * 4 <= max(wl))
            and (len(SENT_RE.findall(bl[wl.index(max(wl))])) >= 3)
        ):
            tags["B:bullet_paragraph"] = "a body paragraph bulleted as an item"
    if kind == "two":
        for l in tail.split("\n"):
            if "******" in l and l.strip() != "******" and (not re.match("\\s*\\*{6}\\s*\\S", l)):
                tags["B:two_inline"] = 1
        parts = [x.strip() for x in tail.split("******")]
        if len(parts) == 2 and all(parts):
            p1, p2 = parts
            lastsent = (units(p1, 0) or [(0, 0, p1)])[-1][2]
            if p1.rstrip("*_ ").endswith(":") or (
                real_words(p1, eng) < 0.4 * real_words(p2, eng)
                and re.search(
                    "(?i)\\b(?:here (?:is|are)|below (?:is|are)|the following)\\b", lastsent
                )
            ):
                tags["B:two_preamble"] = 1
            for p in parts:
                labs = {
                    m.group(1).lower()
                    for m in re.finditer(
                        "(?im)^\\W{0,3}(?:response|answer|option|version|critique|alternative|take)\\s*#?\\s*(1|2|one|two)\\b",
                        p,
                    )
                }
                if len(labs) >= 2:
                    tags["B:two_merged"] = 1
    if plan_.get("title"):
        ts = title_spans(text)
        own = [t for t in ts if t[0] >= start]
        if ts and (not any((title_valid4(text, a, e, inner, eng) for a, e, inner in own))):
            if not own:
                tags["C:title_echo_only"] = 1
            elif not any((valid_title(i) for _, _, i in own)):
                tags["C:title_filler"] = 1
            else:
                tags["C:title_inline"] = 1
    n_ph = plan_.get("placeholders")
    if n_ph:
        ph = ph_official(text)
        good = [p for p in ph if p[0] >= start and ph_valid4(text, p[0], p[1], p[2], eng)]
        if len(ph) >= n_ph and len(good) < n_ph:
            tags["D:ph_fake"] = f"{len(good)}/{len(ph)} real"
    n_hl = plan_.get("highlights")
    if n_hl:
        hs = hl_official(text)
        good = [h for h in hs if h[0] >= start and hl_valid4(text, h[0], h[1], h[2], eng)]
        if len(hs) >= n_hl and len(good) < n_hl:
            tags["E:hl_fake"] = f"{len(good)}/{len(hs)} real"
    if plan_.get("min_words"):
        nw = count_words(text)
        if nw >= plan_["min_words"] and (not is_poemlike(tail)):
            rw = sum((len(WORD_RE.findall(s)) for _, _, s in recycled_units(text, start)))
            if nw - rw < plan_["min_words"] and rw >= 0.15 * nw:
                tags["F:recycled"] = f"{rw} recycled words"
        if len(CONCL.findall(tail)) >= 2:
            tags["F:double_conclusion"] = 1
    if plan_.get("min_sent"):
        if (
            loose_sentence_count(tail) >= plan_["min_sent"]
            and genuine_sentence_count(tail) < plan_["min_sent"]
        ):
            tags["F:sent_fake"] = 1
    for kr in plan_.get("kw_req", ()):
        if kr.get("relation", "at least") != "at least":
            continue
        kw, need = (kr["kw"], int(kr.get("n", 1)))
        official = len(re.findall(re.escape(kw), text, re.I))
        if official >= need:
            n, why = genuine_kw_count(text, kw, start, eng)
            if n < need:
                tags["G:kw_fake"] = f"{kw}: {n}/{need} genuine ({dict(why)})"
    if eng:
        for lr in plan_.get("letters", ()):
            bad = letter_mangled(tail, lr["letter"], lr["relation"])
            if len(bad) >= 2:
                tags["G:letter_mangled"] = f"{len(bad)} words ({lr['letter']})"
    if prefix and (real_words(tail, eng) < 3 or any((k.startswith("I:") for k in tags))):
        tags["H:echo_soup"] = 1
    us = [s for _, _, s in units(tail, 3)]
    meta = [s for s in us if META_RE.search(s)]
    if len(meta) >= 2 and len(meta) >= 0.5 * len(us):
        tags["H:meta_answer"] = f"{len(meta)}/{len(us)}"
    ep = plan_.get("end_phrase")
    if ep:
        t2 = tail.rstrip().rstrip("\"”'").rstrip()
        if t2.lower().endswith(ep.lower()):
            before = t2[: len(t2) - len(ep)]
            sent = re.split("[.!?\\n]", before)[-1]
            if re.search(
                "(?i)\\b(?:last|final|ending|closing) sentence\\b|\\bmy (?:response|answer|reply)\\b|\\b(?:end|conclude|finish)(?:s|ed|ing)? (?:with|by)\\b|\\bwill be\\b|\\bas follows\\b",
                sent,
            ):
                tags["H:end_meta"] = 1
    if plan_.get("postscript"):
        mk = plan_["postscript"]
        pm = list(re.finditer(re.escape(mk).replace("\\ ", "\\s?"), tail))
        if pm:
            after = tail[pm[-1].end() :].strip()
            before = tail[: pm[0].start()]
            if real_words(after, eng) < 2 or after.rstrip().endswith(":"):
                tags["H:ps_dangling"] = 1
            if real_words(before, eng) < 10:
                tags["H:ps_only"] = 1
    if last.rstrip().endswith(":") and (
        not re.match("(?i)^\\s*(?:sincerely|regards|best|yours|thanks|cheers)", last)
    ):
        tags["H:dangling_colon"] = 1
    return tags


_BLOCK_IIDS = (
    "length_constraints:number_paragraphs",
    "detectable_format:multiple_sections",
    "combination:two_responses",
    "detectable_format:number_bullet_lists",
    "length_constraints:nth_paragraph_first_word",
)
G4_RELEVANT = {
    "A:skeleton": _BLOCK_IIDS
    + ("length_constraints:number_words", "length_constraints:number_sentences"),
    "A:fragment_end": _BLOCK_IIDS
    + ("length_constraints:number_words", "length_constraints:number_sentences"),
    "A:empty_block": _BLOCK_IIDS,
    "A:dup_block": _BLOCK_IIDS,
    "B:inline_sep": ("length_constraints:number_paragraphs",),
    "B:sep_mid_list": ("length_constraints:number_paragraphs",),
    "B:sep_mid_sentence": ("length_constraints:number_paragraphs",),
    "B:two_inline": ("combination:two_responses",),
    "B:two_preamble": ("combination:two_responses",),
    "B:two_merged": ("combination:two_responses",),
    "B:bullet_merged": ("detectable_format:number_bullet_lists",),
    "B:bullet_paragraph": ("detectable_format:number_bullet_lists",),
    "C:title_inline": ("detectable_format:title",),
    "C:title_filler": ("detectable_format:title",),
    "C:title_echo_only": ("detectable_format:title",),
    "D:ph_fake": ("detectable_content:number_placeholders",),
    "E:hl_fake": ("detectable_format:number_highlighted_sections",),
    "F:recycled": ("length_constraints:number_words",),
    "F:double_conclusion": ("length_constraints:number_words",),
    "F:sent_fake": ("length_constraints:number_sentences",),
    "G:kw_fake": ("keywords:existence", "keywords:frequency"),
    "G:letter_mangled": ("keywords:letter_frequency",),
    "H:end_meta": ("startend:end_checker",),
    "H:ps_dangling": ("detectable_content:postscript",),
    "H:ps_only": ("detectable_content:postscript",),
    "H:echo_soup": ("combination:repeat_prompt",),
}


def g4_relevant(tags, iid):
    """Tags that bear on one instruction: its own tags plus the generic degenerate / meta / dangling ones."""
    return {
        t: v
        for t, v in tags.items()
        if t.startswith("I:")
        or t in ("H:meta_answer", "H:dangling_colon")
        or iid in G4_RELEVANT.get(t, ())
    }


G4_CONSTRAINT = {
    "A:empty_block": "blocks",
    "A:dup_block": "blocks",
    "B:inline_sep": "number_paragraphs",
    "B:sep_mid_list": "number_paragraphs",
    "B:sep_mid_sentence": "number_paragraphs",
    "B:two_inline": "two_responses",
    "B:two_preamble": "two_responses",
    "B:two_merged": "two_responses",
    "B:bullet_merged": "number_bullet_lists",
    "B:bullet_paragraph": "number_bullet_lists",
    "C:title_inline": "title",
    "C:title_filler": "title",
    "C:title_echo_only": "title",
    "D:ph_fake": "number_placeholders",
    "E:hl_fake": "number_highlighted_sections",
    "F:recycled": "number_words",
    "F:sent_fake": "number_sentences",
    "G:kw_fake": "keywords",
    "G:letter_mangled": "letter_frequency",
    "H:end_meta": "end_checker",
    "H:ps_dangling": "postscript",
    "H:ps_only": "postscript",
    "H:echo_soup": "repeat_prompt",
}


def g4_degenerate(text, plan_=None):
    """I-class tags on a piece of text used to detect degenerate content."""
    return sorted(
        (
            k
            for k in g4_issues(text, dict(language=(plan_ or {}).get("language")), prefix="")
            if k.startswith("I:")
        )
    )


def plan(row):
    p = dict(
        prefix="",
        blocks=None,
        min_words=None,
        max_words=None,
        min_sent=None,
        max_sent=None,
        case=None,
        no_comma=False,
        quotation=False,
        end_phrase=None,
        postscript=None,
        title=False,
        constrained=False,
        placeholders=None,
        highlights=None,
        keywords=[],
        forbidden=[],
        language=None,
        kw_req=[],
        letters=[],
    )
    for iid, kw in zip(row["instruction_id_list"], row["kwargs"]):
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        if iid == "combination:repeat_prompt":
            p["prefix"] = kw["prompt_to_repeat"].strip() + "\n\n" + p["prefix"]
        elif iid == "length_constraints:number_paragraphs":
            p["blocks"] = dict(
                kind="paragraphs",
                n=kw["num_paragraphs"],
                sep="\n***\n",
                block_prefix="",
                cut="\n\n",
            )
        elif iid == "detectable_format:number_bullet_lists":
            p["blocks"] = dict(
                kind="bullets", n=kw["num_bullets"], sep="\n", block_prefix="* ", cut="\n"
            )
        elif iid == "detectable_format:multiple_sections":
            p["blocks"] = dict(
                kind="sections",
                n=kw["num_sections"],
                sep="\n\n",
                block_prefix=kw["section_spliter"].strip() + " {i}\n",
                cut="\n\n",
            )
        elif iid == "length_constraints:nth_paragraph_first_word":
            p["blocks"] = dict(
                kind="nthpara",
                n=kw["num_paragraphs"],
                sep="\n\n",
                block_prefix="",
                cut="\n\n",
                nth=kw["nth_paragraph"],
                first_word=kw["first_word"],
            )
        elif iid == "combination:two_responses":
            p["blocks"] = dict(kind="two", n=2, sep="\n******\n", block_prefix="", cut="******")
        elif iid == "length_constraints:number_words":
            p["min_words" if kw["relation"] == "at least" else "max_words"] = kw["num_words"]
        elif iid == "length_constraints:number_sentences":
            p["min_sent" if kw["relation"] == "at least" else "max_sent"] = kw["num_sentences"]
        elif iid == "change_case:english_capital":
            p["case"] = "upper"
        elif iid == "change_case:english_lowercase":
            p["case"] = "lower"
        elif iid == "punctuation:no_comma":
            p["no_comma"] = True
        elif iid == "startend:quotation":
            p["quotation"] = True
        elif iid == "startend:end_checker":
            p["end_phrase"] = kw["end_phrase"].strip()
        elif iid == "detectable_content:postscript":
            p["postscript"] = kw["postscript_marker"].strip()
        elif iid == "detectable_format:title":
            p["title"] = True
        elif iid == "detectable_format:constrained_response":
            p["constrained"] = True
        elif iid == "detectable_content:number_placeholders":
            p["placeholders"] = kw["num_placeholders"]
        elif iid == "detectable_format:number_highlighted_sections":
            p["highlights"] = kw["num_highlights"]
        elif iid == "keywords:existence":
            p["keywords"] += [w.strip().lower() for w in kw["keywords"] if w.strip()]
            p["kw_req"] += [
                dict(kw=w.strip().lower(), relation="at least", n=1)
                for w in kw["keywords"]
                if w.strip()
            ]
        elif iid == "keywords:frequency":
            p["keywords"].append(kw["keyword"].strip().lower())
            p["kw_req"].append(
                dict(
                    kw=kw["keyword"].strip().lower(),
                    relation=kw.get("relation", "at least"),
                    n=int(kw.get("frequency", 1)),
                )
            )
        elif iid == "keywords:letter_frequency":
            p["letters"].append(
                dict(
                    letter=kw["letter"].strip().lower(),
                    relation=kw.get("let_relation", "at least"),
                    n=int(kw.get("let_frequency", 1)),
                )
            )
        elif iid == "keywords:forbidden_words":
            p["forbidden"] += [w.strip().lower() for w in kw["forbidden_words"] if w.strip()]
        elif iid == "language:response_language":
            p["language"] = kw["language"]
    return p


def plan_slots(p):
    """Format slots a plan controls (for slot-specific filler checks)."""
    s = set()
    if p.get("placeholders"):
        s.add("placeholders")
    if p.get("title"):
        s.add("title")
    if p.get("highlights"):
        s.add("highlights")
    if p.get("postscript"):
        s.add("postscript")
    if p.get("blocks"):
        s.add(p["blocks"]["kind"])
    if p.get("prefix"):
        s.add("repeat_prompt")
    return s


def needs_program3(p):
    """Prompts with a constraint the construction program writes or transforms."""
    return bool(
        p["blocks"]
        or p["prefix"]
        or p["min_words"]
        or p["max_words"]
        or p["min_sent"]
        or p["max_sent"]
        or p["case"]
        or p["no_comma"]
        or p["quotation"]
        or p["end_phrase"]
        or p["postscript"]
        or p["title"]
        or p["constrained"]
        or p["placeholders"]
        or p["highlights"]
    )


G4_GUARD_ALL = os.environ.get("G4_GUARD_ALL", "1") != "0"
G4_NO_GUARDS = os.environ.get("G4_NO_GUARDS", "0") == "1"


def needs_program(p):
    """Constructible prompts plus every other prompt when G4_GUARD_ALL is on. Other prompts keep their native
    text (body path) and only pass through the guarded certification, so certification-made loopholes on lexical
    constraints (mangled words for letter constraints, glued / split keywords) are rejected as well."""
    return needs_program3(p) or G4_GUARD_ALL


BLOCK_IID = dict(
    paragraphs="length_constraints:number_paragraphs",
    bullets="detectable_format:number_bullet_lists",
    sections="detectable_format:multiple_sections",
    nthpara="length_constraints:nth_paragraph_first_word",
    two="combination:two_responses",
)


def block_kwargs(b):
    if b["kind"] == "paragraphs":
        return dict(num_paragraphs=b["n"])
    if b["kind"] == "bullets":
        return dict(num_bullets=b["n"])
    if b["kind"] == "sections":
        return dict(section_spliter=b["block_prefix"].split(" {i}")[0].strip(), num_sections=b["n"])
    if b["kind"] == "nthpara":
        return dict(num_paragraphs=b["n"], nth_paragraph=b["nth"], first_word=b["first_word"])
    return {}


def native_has_blocks(b, text):
    """Native blocks must not be filler, duplicates, or shorter than the hardened
    minimum; hardened predicates decide when available, with base predicates as fallback."""
    import pace.constraints.base as PV

    if _PH is not None:
        try:
            if not all(
                (d.accepts_text(text) for d in _PH.predicate(BLOCK_IID[b["kind"]], block_kwargs(b)))
            ):
                return False
        except Exception:
            pass
    try:
        if b["kind"] == "paragraphs":
            ok = PV.paragraphs(b["n"]).accepts_text(text)
            parts = re.split("\\s?\\*\\*\\*\\s?", text)
        elif b["kind"] == "bullets":
            ok = PV.bullets(b["n"]).accepts_text(text)
            parts = [l for l in text.splitlines() if re.match("^\\s*[*-]\\s", l)]
        elif b["kind"] == "sections":
            sp = b["block_prefix"].split(" {i}")[0].strip()
            ok = PV.sections(sp, b["n"]).accepts_text(text)
            parts = re.split("\\s?" + re.escape(sp) + "\\s?\\d+\\s?", text)[1:]
        elif b["kind"] == "nthpara":
            ok = PV.nth_paragraph_first_word(b["n"], b["nth"], b["first_word"]).accepts_text(text)
            parts = re.split("\\n\\s*\\n", text)
        elif b["kind"] == "two":
            parts = text.split("******")
            ok = len(parts) == 2 and all((x.strip() for x in parts))
        else:
            return False
        if not ok:
            return False
        bs = [re.sub("^\\s*[*-]\\s", "", x).strip() for x in parts if x.strip()]
        mw = BULLET_MIN_WORDS if b["kind"] == "bullets" else BLOCK_MIN_WORDS
        if any((len(WORD_RE.findall(x)) < mw or FILLER_LINE.match(x) for x in bs)):
            return False
        for i in range(len(bs)):
            if any((near_dup(bs[i], bs[j]) for j in range(i))):
                return False
        return True
    except Exception:
        return False


def cut_block(text, cut):
    text = text.lstrip()
    text = (
        re.sub(
            "^(?:\\s*(?:\\*{3,}|\\*{6}|\\*+\\s*\\d*\\s*[:.)]?\\s|\\* |- |\\d+\\s*[:.)]\\s)\\s*)+",
            "",
            text,
        ).lstrip()
        if cut is not None
        else text
    )
    if cut is None:
        return text.strip()
    i = text.find(cut)
    return (text if i < 0 else text[:i]).strip()


def block_end(text, b):
    """Where a generated block ends: the plan's cut string, the model's own '***' separator, the next section header,
    or a single-newline paragraph break after one complete prose line (>= 20 words ending in . ! ? and followed by a
    capitalised line). Without this, one block swallows the model's remaining paragraphs and later blocks come out
    empty, leaving doubled separators for certification to 'repair' with junk."""
    ends = []
    if b.get("cut"):
        i = text.find(b["cut"])
        if i >= 0:
            ends.append(i)
    if b["kind"] in ("paragraphs", "nthpara", "sections", "two"):
        m = re.search("\\n[ \\t]*\\*{3,}[ \\t]*(?:\\n|$)", text)
        if m:
            ends.append(m.start())
        if b["kind"] == "sections":
            sp = b["block_prefix"].split(" {i}")[0].strip()
            m = re.search("\\n[ \\t]*" + re.escape(sp) + "\\s*\\d", text, re.I)
            if m:
                ends.append(m.start())
        if b["kind"] in ("paragraphs", "nthpara"):
            m = re.search('[.!?]["\\\')]?\\n(?=[A-Z"*\\[])', text)
            if (
                m
                and "\n" not in text[: m.start()]
                and (len(WORD_RE.findall(text[: m.start()])) >= 20)
            ):
                ends.append(m.start() + 1)
    return min(ends) if ends else -1


def cut_block_plan(text, b):
    """cut_block with the plan-aware block end, trailing separators removed."""
    text = text.lstrip()
    text = re.sub(
        "^(?:\\s*(?:\\*{3,}|\\*{6}|\\*+\\s*\\d*\\s*[:.)]?\\s|\\* |- |\\d+\\s*[:.)]\\s)\\s*)+",
        "",
        text,
    ).lstrip()
    e = block_end(text, b)
    cand = text if e < 0 else text[:e]
    cand = re.sub("(?:\\s*\\n[ \\t]*\\*{2,}[ \\t]*)+$", "", cand)
    return cand.strip()


def truncate_words(text, max_words):
    words = list(WORD_RE.finditer(text))
    if len(words) < max_words:
        return text
    head = text[: words[max_words - 1].start()]
    m = list(SENT_RE.finditer(head))
    if m and m[-1].end() > len(head) * 0.5:
        return head[: m[-1].end()].rstrip()
    return head.rstrip().rstrip(",;:") + "."


def truncate_sentences(text, max_sent):
    ends = list(SENT_RE.finditer(text))
    if len(ends) < max_sent:
        return text
    return text[: ends[max(0, max_sent - 2)].end()].rstrip()


def sentence_end_before(text, pos):
    """End offset of the last sentence/line ending at or before pos (0 if none)."""
    ends = [m.end() for m in SENT_RE.finditer(text) if m.end() <= pos] + [
        m.start() for m in re.finditer("\\n", text) if m.start() <= pos
    ]
    return max(ends) if ends else 0


def add_highlights(body, need, start_at=0):
    """Exact transform: wrap `need` real sentences (>= 3 words, distinct lines, not bullets / titles / headers /
    separators / lines with asterisks, never before start_at = end of a clamped prefix) in *...*. Returns (body, literals)."""
    cands_by_line = []
    for lm in re.finditer("[^\\n]+", body):
        line = lm.group()
        if lm.start() < start_at:
            continue
        if (
            "*" in line
            or line.lstrip().startswith(("#", "<<", "- ", ">", "|", "```"))
            or re.match(
                "^\\s*(?:SECTION|Section|PARAGRAPH|Paragraph|Part|PART|Chapter|CHAPTER)\\s*\\d+",
                line,
            )
            or re.fullmatch("\\s*\\W*\\s*", line)
            or re.match("^\\s*\\d+[.)]\\s*$", line)
        ):
            continue
        cands, start = ([], 0)
        for sm in SENT_RE.finditer(line):
            seg = line[start : sm.end()]
            if len(WORD_RE.findall(seg)) >= 3 and (not re.search("[\\[\\]<>{}]", seg)):
                cands.append((lm.start() + start, lm.start() + sm.end()))
            start = sm.end()
        rest = line[start:]
        if len(WORD_RE.findall(rest)) >= 3 and (not re.search("[\\[\\]<>{}]", rest)):
            cands.append((lm.start() + start, lm.start() + len(line)))
        if cands:
            cands_by_line.append(cands)
    chosen, r = ([], 0)
    while len(chosen) < need and any(cands_by_line):
        for cands in cands_by_line:
            if len(chosen) >= need:
                break
            if cands:
                chosen.append(cands.pop(0))
    if len(chosen) < need:
        for lm in re.finditer("(?m)^\\s*\\* (.+)$", body):
            if len(chosen) >= need:
                break
            if lm.start() < start_at or "*" in lm.group(1) or len(WORD_RE.findall(lm.group(1))) < 3:
                continue
            chosen.append((lm.start(1), lm.end(1), "**"))
    chosen.sort()
    out, pos, lits = ([], 0, [])
    for c in chosen:
        a, b = (c[0], c[1])
        star = c[2] if len(c) > 2 else "*"
        seg = body[a:b]
        lead = seg[: len(seg) - len(seg.lstrip())]
        core = seg.strip()
        trail = seg[len(seg.rstrip()) :]
        out.append(body[pos:a])
        out.append(lead + star + core + star + trail)
        pos = b
        lits.append(star + core + star)
    out.append(body[pos:])
    return ("".join(out), lits)


def g4_clean_block(cand, b):
    """Generic hygiene for a generated block: drop bare list markers / empty label items; a block never carries a
    separator of its own (cut at an inline '***' / '******'); strip an enumerator the model wrote inside a bullet
    ('3 Compatibility ...') and cut a bullet where a second enumerated item starts ('... long-lasting 2 Sleek')."""
    kind = b["kind"]
    c = strip_skeleton(cand)
    if kind in ("paragraphs", "two", "nthpara", "sections"):
        m = re.search("\\*{3,}", c)
        if m:
            c = c[: m.start()]
    if kind == "bullets":
        c = re.sub("^\\s*(?:\\d{1,2}[.)]?|[*\\-•])\\s+(?=\\S)", "", c)
        m = re.search("(?<=\\w)[ \\t]+\\d{1,2}[.)]?[ \\t]+(?=[A-Z][a-z])", c)
        if m:
            c = c[: m.start()]
    c = re.sub("[ \\t]+\\n", "\n", c)
    c = re.sub("^[\\s.,;:!?]+(?=\\w)", "", c)
    return c.strip()


TWO_LABELS = re.compile(
    "(?im)^\\W{0,3}(?:response|answer|option|version|critique|alternative|take)\\s*#?\\s*(1|2|one|two)\\b"
)
PREAMBLE_END = re.compile("(?i)\\b(?:here (?:is|are)|below (?:is|are)|the following)\\b[^.!?]*$")


def g4_piece_issues(cand, b, i, plan_):
    """A/B/H/I tags on one generated block (before it is accepted)."""
    tags = set(g4_degenerate(cand, plan_))
    if b["kind"] == "two":
        if i == 1 and (cand.rstrip("*_ ").endswith(":") or PREAMBLE_END.search(cand)):
            tags.add("B:two_preamble")
        if len({m.group(1).lower() for m in TWO_LABELS.finditer(cand)}) >= 2:
            tags.add("B:two_merged")
    bare, empty_lab, _ = skeleton_lines(cand)
    if len(bare) >= 2 or len(empty_lab) >= 3:
        tags.add("A:skeleton")
    if cand.rstrip("*_ ").endswith(":"):
        tags.add("H:dangling_colon")
    return sorted(tags)


def split_conclusion(body, start=0):
    """(head, conclusion) when the body ends with a conclusion ('In conclusion, ...') that covers <= 40% of its
    words; (body, '') otherwise. Continuations are generated before the conclusion, never after it."""
    ms = [m for m in CONCL.finditer(body) if m.start() >= start]
    if not ms:
        return (body, "")
    a = ms[-1].start()
    if count_words(body[a:]) > 0.4 * count_words(body[start:]) or count_words(body[start:a]) < 30:
        return (body, "")
    return (body[:a].rstrip(), body[a:].lstrip())


def drop_recycled(cont, body):
    """Remove the units of a continuation that recycle the body (4-gram containment or Jaccard >= 0.6)."""
    joined = body.rstrip() + "\n\n"
    full = joined + cont
    rec = recycled_units(full, start=len(joined))
    if not rec:
        return (cont, 0)
    out, pos = ([], len(joined))
    for a, e, _ in rec:
        if a < pos:
            continue
        out.append(full[pos:a])
        pos = e
    out.append(full[pos:])
    t = "".join(out)
    t = re.sub("[ \\t]{2,}", " ", t)
    t = re.sub(" +\\n", "\n", t)
    t = re.sub("\\n{3,}", "\n\n", t)
    return (t.strip(), len(rec))


def strip_plan_markers(cont, p):
    """A length continuation may not add structure the plan counts (bullets, '***', '******', section headers,
    paragraphs of an nth-paragraph plan, a P.S., a title)."""
    b = p.get("blocks") or {}
    kind = b.get("kind")
    sp = b.get("block_prefix", "").split(" {i}")[0].strip() if kind == "sections" else None
    out = []
    for l in cont.split("\n"):
        if kind == "bullets" and (re.match("^\\s*\\*[^*]", l) or re.match("^\\s*-", l)):
            continue
        if sp and re.match("^\\s*" + re.escape(sp) + "\\s?\\d", l, re.I):
            break
        if p.get("postscript") and re.match("(?i)^\\s*p\\.\\s?p?\\.?\\s?s\\b", l):
            break
        out.append(l)
    c = "\n".join(out)
    if kind == "paragraphs":
        c = re.split("\\*{3,}", c)[0]
    if kind == "two":
        c = c.split("******")[0]
    if kind == "nthpara":
        c = re.split("\\n\\s*\\n", c)[0]
    if p.get("title"):
        c = re.sub("<<[^\\n]*?>>", "", c)
    return c.strip()


def strip_meta_highlights(body, start=0):
    """Remove highlight tokens whose content is a meta label ('*Highlighted section Part 2*', '**Light**'); real
    highlights are then added as an exact transform on real sentences."""
    out, pos, n = ([], 0, 0)
    for a, e, inner in sorted(hl_official(body)):
        if a < pos or a < start:
            continue
        if META_HL.match(inner) or "highlighted section" in inner.lower():
            out.append(body[pos:a])
            pos = e
            n += 1
            m = re.match("[ \\t]*:?[ \\t]*", body[pos:])
            pos += m.end() if m else 0
    if not n:
        return (body, 0)
    out.append(body[pos:])
    t = "".join(out)
    t = re.sub("(?m)^[ \\t]+$", "", t)
    t = re.sub("\\n{3,}", "\n\n", t)
    return (t.strip("\n"), n)


def valid_highlight_count(body, start=0, english=True):
    return sum(
        (
            1
            for a, e, inner in hl_official(body)
            if a >= start and hl_valid4(body, a, e, inner, english)
        )
    )


def fix_end_meta(text, ep, start=0):
    """End phrase quoted inside a final meta-statement ('the last sentence of my response will be "<phrase>"'):
    drop that sentence so that the phrase is appended as its own last sentence."""
    t2 = text.rstrip()
    idx = t2.lower().rfind(ep.lower())
    if idx < start:
        return (text, False)
    cut = sentence_end_before(t2, idx)
    if cut <= start:
        return (text, False)
    return (t2[:cut].rstrip(), True)


def drop_dangling_intro(body, start=0):
    """A final line that only announces content ('Here are 10 names for a boy:', a bare '1.') is removed when at
    least 15 words of real content precede it."""
    lines = body.rstrip().split("\n")
    last = lines[-1] if lines else ""
    if (
        last.rstrip("*_ ").endswith(":")
        and count_words(last) <= 30
        or (BARE_MARK.match(last) and last.strip() not in ("***", "******"))
    ):
        head = "\n".join(lines[:-1]).rstrip()
        if len(head) > start and count_words(head[start:]) >= 15:
            return (head, True)
    return (body, False)


def title_candidate_ok(cand, english=True):
    """A generated title: 2-12 words, title-like (capitalised start), mostly English words, no splices."""
    if not (
        valid_title(cand)
        and 2 <= len(cand.split()) <= 12
        and (len(ALPHA_RE.findall(cand)) >= 2)
        and (not splice_tags(cand))
    ):
        return False
    return title_valid4("<<" + cand + ">>", 0, len(cand) + 4, cand, english)


G4_NATIVE_BLOCK_VETO = frozenset(
    (
        "A:skeleton",
        "A:empty_block",
        "A:dup_block",
        "B:inline_sep",
        "B:sep_mid_list",
        "B:sep_mid_sentence",
        "B:two_inline",
        "B:two_preamble",
        "B:two_merged",
        "B:bullet_merged",
        "B:bullet_paragraph",
        "I:pseudo_words",
        "I:pseudo_script",
        "I:letter_run",
        "I:symbol_run",
    )
)


class Program:
    def __init__(self, T, re_t=0, max_rounds=3, seed=0, gen_total=1280, max_tries=3, g4_extra=None):
        self.T, self.re_t, self.max_rounds, self.seed, self.gen_total, self.max_tries = (
            T,
            re_t,
            max_rounds,
            seed,
            gen_total,
            max_tries,
        )
        self.tok = T.tok
        self.g4_cap = (
            int(os.environ.get("G4_MAX_GEN_CALLS", 14)) if g4_extra is None else int(g4_extra)
        )
        self.ncalls = 0
        try:
            self._params = set(inspect.signature(T.generate_suffix).parameters)
        except (TypeError, ValueError):
            self._params = set()

    def enc(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def gen(self, prompt, prefix_text, seed, attempt=0):
        """Suffix generation after a fixed answer prefix. attempt>0 changes the seed AND the canvas (gen_length /
        total), since masked-diffusion argmax samplers (LLaDA, Dream) are seed-independent. Returns the decoded
        continuation (free positions only; no prefix round-trip) and the seconds spent."""
        pre = self.enc(prefix_text) if prefix_text else []
        kw = {}
        if "total" in self._params and self.gen_total < 10**8:
            total = self.gen_total if len(pre) + 200 < self.gen_total - 200 else 2048
            if attempt:
                total = min(2048, total + 128 * attempt)
            kw["total"] = total
        if attempt and "gen_length" in self._params:
            kw["gen_length"] = (256, 320, 192, 384)[attempt % 4]
        out = self.T.generate_suffix(prompt, pre, seed + 1000 * attempt, **kw)
        self.ncalls += 1
        ids = list(out["ids"])
        fs = int(out.get("free_start", len(pre)))
        cont = self.T.decode(ids[fs:])[0]
        cont = cont.replace("\r\n", "\n").replace("\r", "\n")
        return (cont, float(out.get("seconds", 0.0)))

    def g4(self, msg, log, tag=None):
        """Record a comparative guard firing (log line + weak tag, both 'g4:'-prefixed)."""
        log.append("g4: " + msg)
        if tag and "g4:" + tag not in self.weak:
            self.weak.append("g4:" + tag)

    def budget_left(self):
        return self.ncalls < self.g4_cap

    def _block(self, prompt, body, bp, b, i, prev, log, p):
        """One block: generate, cut, clean with structural cleanup; regenerate when empty / filler / duplicate / label-glued /
        degenerate / preamble / skeleton."""
        first = b.get("first_word") if b["kind"] == "nthpara" and i == b["nth"] else None
        last_cand = ""
        for t in range(self.max_tries):
            cont, s = self.gen(prompt, body + bp, self.seed + i, attempt=t)
            self.cost += s
            cand = cut_block_plan(cont, b)
            cand = clean_piece(cand, body)
            c4 = g4_clean_block(cand, b)
            if c4 != cand:
                self.g4(
                    f"block {i}: hygiene (skeleton / inline separator / enumerator) try {t}",
                    log,
                    "A:block_hygiene",
                )
                cand = c4
            if first and (
                re.match("^\\s*\\d+\\s*[:.)]", cont.lstrip())
                or re.match("^\\s*\\d+\\s*[:.)]", cand)
            ):
                log.append(f"block {i}: label glue regenerated")
                self.regens += 1
                last_cand = cand
                continue
            mw = BULLET_MIN_WORDS if b["kind"] == "bullets" else BLOCK_MIN_WORDS
            if not cand or len(WORD_RE.findall(cand)) < mw or FILLER_LINE.match(cand):
                log.append(f"block {i}: empty/filler/short try {t}")
                self.regens += 1
                continue
            if any((near_dup(cand, pb) for pb in prev)):
                log.append(f"block {i}: duplicate try {t}")
                self.regens += 1
                continue
            gc = _grams(cand)
            if (
                not G4_NO_GUARDS
                and len(gc) >= 8
                and any((len(gc & _grams(pb)) / len(gc) >= 0.5 for pb in prev))
            ):
                self.g4(f"block {i}: mostly copies an earlier block try {t}", log, "A:dup_block")
                self.regens += 1
                continue
            if is_degenerate(cand):
                cand2 = drop_repeats(cand)
                if (
                    len(WORD_RE.findall(cand2)) >= mw
                    and (not any((near_dup(cand2, pb) for pb in prev)))
                    and (not g4_piece_issues(cand2, b, i, p))
                ):
                    log.append(f"block {i}: repeats dropped")
                    return cand2
                log.append(f"block {i}: degenerate try {t}")
                self.regens += 1
                continue
            iss = g4_piece_issues(cand, b, i, p)
            if iss:
                self.g4(f"block {i}: {iss} try {t}", log, iss[0])
                self.regens += 1
                continue
            return cand
        if (
            first
            and last_cand
            and (len(WORD_RE.findall(last_cand)) >= BLOCK_MIN_WORDS)
            and (not any((near_dup(last_cand, pb) for pb in prev)))
        ):
            self.weak.append("nth_paragraph_label")
            return last_cand
        return ""

    def _plain_body(self, prompt, prefix, native_text, log, use_native=True):
        """Construct a body without a block plan, rejecting content flagged by the I-class guards."""
        pn = norm(prefix)
        if (
            use_native
            and (not prefix)
            and (native_text is not None)
            and (not is_degenerate(native_text))
        ):
            d4 = g4_degenerate(native_text, self._plan) + (
                ["H:meta_answer"]
                if "H:meta_answer" in g4_issues(native_text, {}, prefix="")
                else []
            )
            if not d4:
                return native_text
            self.g4(f"native body degenerate {d4} -> regenerate", log, d4[0])
        best, best_score = (None, None)
        for t in range(self.max_tries):
            if t and (not self.budget_left()):
                self.g4("body: budget exhausted", log, "budget")
                break
            attempt = t + (1 if native_text is not None and (not prefix) else 0)
            cont, s = self.gen(prompt, prefix, self.seed, attempt=attempt)
            self.cost += s
            cand = prefix + clean_piece(cont.lstrip(), prefix) if prefix else cont.strip()
            tail = cand[len(prefix) :]
            d4 = g4_degenerate(tail, self._plan) + (
                ["H:meta_answer"] if "H:meta_answer" in g4_issues(tail, {}, prefix="") else []
            )
            bad = (
                is_degenerate(tail)
                or bool(d4)
                or (
                    prefix
                    and (echo_only(tail, pn) or len(WORD_RE.findall(tail)) < REPEAT_MIN_WORDS)
                )
                or (len(WORD_RE.findall(tail)) < 3)
            )
            if not bad:
                return cand
            score = (
                sum((b - a for a, b, _ in repeated_spans(tail)))
                + loop_score(tail) * len(tail)
                - min(len(WORD_RE.findall(tail)), REPEAT_MIN_WORDS)
                + 200 * len(d4)
            )
            if best is None or score < best_score:
                best, best_score = (cand, score)
            log.append(f"body: degenerate/echo try {t}" + (f" g4 {d4}" if d4 else ""))
            self.regens += 1
        if best is None:
            return native_text if native_text is not None else prefix
        body = prefix + drop_repeats(collapse_loops(best[len(prefix) :]), prefix)
        tail = body[len(prefix) :]
        if (
            native_text is not None
            and (not prefix)
            and g4_degenerate(tail, self._plan)
            and (not is_degenerate(native_text))
            and (
                len(g4_degenerate(native_text, self._plan)) <= len(g4_degenerate(tail, self._plan))
            )
        ):
            self.g4("regenerations no better than native -> native kept", log, "fallback_native")
            return native_text
        self.weak.append(
            "short_answer"
            if prefix and (not is_degenerate(tail)) and (not echo_only(tail, pn))
            else "degenerate_body"
        )
        log.append("body kept with repeats dropped")
        return body

    def _continue(self, prompt, body, p, log):
        """Length budget (<= max_rounds rounds). Continuations are generated BEFORE a final conclusion (which is
        re-appended), recycled units are dropped, the continuation may not add counted structure, and a list /
        heading ending gets a paragraph break instead of being extended in-line."""
        rounds = 0
        start = len(p["prefix"]) if p["prefix"] and body.startswith(p["prefix"]) else 0
        kind = (p.get("blocks") or {}).get("kind")

        def short(t):
            return (
                p["min_words"]
                and count_words(t) < p["min_words"]
                or (p["min_sent"] and count_sent(t) < p["min_sent"])
            )

        while rounds < self.max_rounds and short(body):
            if rounds and (not self.budget_left()):
                self.g4("continue: budget exhausted", log, "budget")
                break
            head, concl = split_conclusion(body, start) if kind != "nthpara" else (body, "")
            last_line = (head.rstrip().split("\n") or [""])[-1]
            if concl:
                pre = head.rstrip() + "\n\n"
            elif kind != "nthpara" and (
                LIST_ITEM.match(last_line) or kind == "bullets" or last_line.rstrip().endswith(":")
            ):
                pre = head.rstrip() + "\n\n"
            else:
                pre = head.rstrip() + " "
            cont, s = self.gen(prompt, pre, self.seed + 50 + rounds, attempt=rounds)
            self.cost += s
            rounds += 1
            cont = clean_piece(cont.strip(), body)
            cont, nrec = drop_recycled(cont, body)
            if nrec:
                self.g4(
                    f"continue round {rounds}: {nrec} recycled units dropped",
                    log,
                    "F:recycled_dropped",
                )
            c2 = strip_plan_markers(cont, p)
            if c2 != cont:
                self.g4(
                    f"continue round {rounds}: counted structure removed from continuation",
                    log,
                    "B:continuation_structure",
                )
                cont = c2
            if concl or CONCL.search(body[start:]):
                m = CONCL.search(cont)
                if m and (not G4_NO_GUARDS):
                    cont = cont[: m.start()].rstrip()
                    self.g4(
                        f"continue round {rounds}: second conclusion cut",
                        log,
                        "F:second_conclusion",
                    )
            d4 = g4_degenerate(cont, p) if len(WORD_RE.findall(cont)) >= 3 else []
            if d4:
                self.g4(f"continue round {rounds}: degenerate {d4}", log, d4[0])
                self.regens += 1
                continue
            if len(WORD_RE.findall(cont)) < 3:
                log.append(f"continue round {rounds}: nothing new")
                self.regens += 1
                continue
            if concl:
                body = head.rstrip() + "\n\n" + cont.strip() + "\n\n" + concl
            else:
                body = pre + cont.strip()
        if rounds:
            log.append(
                f"continued x{rounds} -> {count_words(body)} words / {count_sent(body)} sentences"
            )
        if short(body):
            if p["min_words"] and count_words(body) < p["min_words"]:
                self.unmet.append("number_words")
            if p["min_sent"] and count_sent(body) < p["min_sent"]:
                self.unmet.append("number_sentences")
            log.append("length budget unmet (no filler)")
        return body

    def _placeholders(self, prompt, body, n, log):
        eng = is_english(body, self._plan)
        have = sum((1 for a, e, inner in ph_official(body) if ph_valid4(body, a, e, inner, eng)))
        if have >= n:
            return body
        for t in range(2):
            if have >= n:
                break
            if t and (not self.budget_left()):
                self.g4("placeholders: budget exhausted", log, "budget")
                break
            cont, s = self.gen(prompt, body.rstrip() + "\n\n[", self.seed + 30 + t, attempt=t)
            self.cost += s
            seg = "[" + cont
            seg = seg[: seg.find("\n\n")] if "\n\n" in seg else seg
            seg = clean_piece(seg, body)
            allph = placeholders_in(seg)
            ph = [m for m in allph if ph_valid4(seg, m.start(), m.end(), m.group(1), eng)]
            if len(ph) < len(allph):
                self.g4(
                    f"placeholders: {len(allph) - len(ph)} invalid bracket(s) in try {t}",
                    log,
                    "D:ph_invalid",
                )
            if not ph:
                log.append(f"placeholders: none generated try {t}")
                self.regens += 1
                continue
            k = min(len(ph), n - have)
            end = ph[k - 1].end()
            m = re.compile("[.!?]+(?=\\s|$)|\\n").search(seg, end)
            seg = seg[: m.end() if m else len(seg)].rstrip()
            if any(
                (
                    not ph_valid4(seg, x.start(), x.end(), x.group(1), eng)
                    for x in placeholders_in(seg)
                )
            ):
                self.g4(
                    f"placeholders: segment keeps an invalid bracket try {t}", log, "D:ph_invalid"
                )
                self.regens += 1
                continue
            if filler_tags(seg) or is_degenerate(seg, body) or g4_degenerate(seg, self._plan):
                log.append(f"placeholders: junk segment try {t}")
                self.regens += 1
                continue
            body = body.rstrip() + "\n\n" + seg
            have += k
            log.append(f"placeholders: +{k} generated")
        if have < n:
            self.unmet.append("number_placeholders")
            log.append(f"placeholders unmet ({have}/{n})")
        return body

    def _title(self, prompt, body, log):
        eng = is_english(body, self._plan)
        fl, _, rest0 = body.partition("\n")
        flc = fl.strip().strip("#* \"'")
        if (
            flc
            and 2 <= len(flc.split()) <= 12
            and (len(ALPHA_RE.findall(flc)) >= 2)
            and (not re.search("[.!?,;:]$", flc))
            and rest0.startswith("\n")
            and rest0.strip()
            and valid_title(flc)
            and title_candidate_ok(flc, eng)
        ):
            log.append("title from heading")
            return (flc, rest0.lstrip())
        leads = [("<<", 0), ("Title: ", 1), (body.rstrip() + "\n\nTitle: ", 2)]
        for t, (lead, att) in enumerate(leads):
            if t == 2 and (not self.budget_left()):
                self.g4("title: budget exhausted", log, "budget")
                break
            cont, s = self.gen(prompt, lead, self.seed + 7 + t, attempt=att)
            self.cost += s
            first = cont.split("\n")[0]
            j = first.find(">>")
            cand = (
                first[:j]
                if 0 <= j < 80
                else first
                if len(first.split()) <= 12 and (not re.search("[.,;:]\\s*$", first))
                else ""
            )
            cand = re.sub("</?title>?", " ", cand, flags=re.I)
            cand = " ".join(cand.strip().strip("<> \"'*#").split())
            if title_candidate_ok(cand, eng):
                log.append("title generated" + (" (body-conditioned)" if t == 2 else ""))
                return (cand, body)
            log.append(f"title: invalid candidate try {t} ({cand[:30]!r})")
            self.regens += 1
            if t == 2:
                self.g4("title: body-conditioned try invalid too", log, "C:title_unmet")
        self.unmet.append("title")
        log.append("title unmet")
        return ("", body)

    def _postscript(self, prompt, body, marker, log):
        for t in range(2):
            pre = body.rstrip() + "\n\n" + marker + " "
            cont, s = self.gen(prompt, pre, self.seed + 9 + t, attempt=t)
            self.cost += s
            cand = clean_piece(cut_block(cont, "\n\n"), body)
            ok3 = (
                len(WORD_RE.findall(cand)) >= POSTSCRIPT_MIN_WORDS
                and (not FILLER_LINE.match(cand))
                and (not filler_tags(cand))
            )
            if ok3 and G4_NO_GUARDS:
                log.append("postscript generated")
                return marker + " " + cand
            if ok3:
                why = []
                if cand.rstrip("*_ ").endswith(":") or PREAMBLE_END.search(cand):
                    why.append("H:ps_dangling")
                if META_RE.search(cand):
                    why.append("H:meta_answer")
                why += g4_degenerate(cand, self._plan)
                if why:
                    self.g4(f"postscript: {why} try {t}", log, why[0])
                    self.regens += 1
                    continue
                log.append("postscript generated")
                return marker + " " + cand
            log.append(f"postscript: invalid try {t}")
            self.regens += 1
        self.unmet.append("postscript")
        log.append("postscript unmet")
        return ""

    def run(self, row, native_text=None):
        p = plan(row)
        t0 = time.perf_counter()
        self.cost = 0.0
        self.regens = 0
        self.unmet = []
        self.weak = []
        log = []
        self._plan = p
        self.ncalls = 0
        prompt = row["prompt"]
        prefix = p["prefix"]
        lits = []
        if prefix:
            lits.append(prefix)
        eng_plan = p.get("language") in (None, "en")
        if p["blocks"] is None:
            body = self._plain_body(prompt, prefix, native_text, log)
        elif (
            native_text is not None
            and (not prefix or native_text.startswith(prefix))
            and native_has_blocks(p["blocks"], native_text)
            and (not is_degenerate(native_text))
            and (not filler_tags(native_text) - {"meta_note"})
            and (not set(g4_issues(native_text, p)) & G4_NATIVE_BLOCK_VETO)
        ):
            body = native_text
            log.append(f"blocks={p['blocks']['kind']} kept from native")
        else:
            if native_text is not None and native_has_blocks(p["blocks"], native_text):
                v = sorted(set(g4_issues(native_text, p)) & G4_NATIVE_BLOCK_VETO)
                if v:
                    self.g4(f"native blocks rejected {v}", log, v[0])
            b = p["blocks"]
            body = prefix
            prev = []
            built = 0
            for i in range(1, b["n"] + 1):
                bp = b["block_prefix"].format(i=i)
                if b["kind"] == "nthpara" and i == b["nth"]:
                    bp = b["first_word"][:1].upper() + b["first_word"][1:] + " "
                block = self._block(prompt, body, bp, b, i, prev, log, p)
                if not block:
                    log.append(f"block {i}: no genuine content -> stop (no filler)")
                    break
                if bp:
                    lits.append(bp.rstrip() if bp.endswith(" ") else bp)
                body += bp + block
                prev.append(block)
                built += 1
                if i < b["n"]:
                    body += b["sep"]
                    lits.append(b["sep"].strip() or b["sep"])
            if built < b["n"]:
                if body.endswith(b["sep"]):
                    body = body[: -len(b["sep"])]
                    lits.pop()
                self.unmet.append(BLOCK_UNMET[b["kind"]])
            log.append(f"blocks={b['kind']} n={built}/{b['n']}")
            if built == 0:
                lits = [prefix] if prefix else []
                body = self._plain_body(
                    prompt,
                    prefix,
                    native_text if not prefix or (native_text or "").startswith(prefix) else None,
                    log,
                )
                if prefix and (not body.startswith(prefix)):
                    body = prefix + body.lstrip()
                self.g4("no block built -> plain answer kept, structure unmet", log, "A:no_blocks")
        body = self._continue(prompt, body, p, log)
        start = len(prefix) if prefix and body.startswith(prefix) else 0
        sk = strip_skeleton(body[start:]) if not G4_NO_GUARDS else body[start:].strip("\n")
        if sk != body[start:].strip("\n") and sk.strip():
            self.g4("skeleton lines (bare list markers / empty labels) removed", log, "A:skeleton")
            body = body[:start] + sk
        body2, dropped = drop_dangling_intro(body, start)
        if dropped:
            self.g4("dangling intro line removed", log, "H:dangling_colon")
            body = body2
        d4 = g4_degenerate(body[start:], p)
        if (
            d4
            and native_text is not None
            and (not prefix or native_text.startswith(prefix))
            and (not is_degenerate(native_text))
            and (not g4_degenerate(native_text[start:], p))
            and (body != native_text)
        ):
            self.g4(f"program body degenerate {d4} -> native body", log, "fallback_native")
            body = native_text
            if p["blocks"] and (not native_has_blocks(p["blocks"], native_text)):
                self.unmet.append(BLOCK_UNMET[p["blocks"]["kind"]])
            lits = [prefix] if prefix else []
        if p["placeholders"]:
            body = self._placeholders(prompt, body, p["placeholders"], log)
        title, ps, opt = ("", "", "")
        hl = len(prefix) if prefix and body.startswith(prefix) else 0
        eng = is_english(body[hl:], p)
        if p["title"] and (
            not any(
                (
                    a >= hl and title_valid4(body, a, e, inner, eng)
                    for a, e, inner in title_spans(body)
                )
            )
        ):
            tail = body[hl:]
            if re.search("<<[^\\n<>]*>>", tail):
                tail = (
                    re.sub("(?m)^(?:title:\\s*)?<<[^\\n<>]*>>[ \\t]*\\n*", "", tail, count=1)
                    if re.match("\\s*(?:title:\\s*)?<<[^\\n<>]*>>\\s*\\n", tail, re.I)
                    else re.sub("<<[^\\n<>]*>>[ \\t]*", "", tail, count=1)
                )
                tail = re.sub("[ \\t]{2,}", " ", tail).strip()
            title, tail = self._title(prompt, tail, log)
            body = body[:hl] + tail
        if p["postscript"] and "H:ps_dangling" in g4_issues(
            body, dict(postscript=p["postscript"], language=p.get("language")), prefix=body[:hl]
        ):
            ms = list(re.finditer(re.escape(p["postscript"]).replace("\\ ", "\\s?"), body[hl:]))
            if ms:
                a = hl + ms[-1].start()
                after = body[hl + ms[-1].end() :]
                if count_words(after) <= 30:
                    body = body[:a].rstrip()
                    self.g4(
                        "empty / dangling postscript removed (a real one is generated)",
                        log,
                        "H:ps_dangling",
                    )
        if p["postscript"] and (
            not re.search(
                "p\\.\\s?p\\.\\s?s" if p["postscript"] == "P.P.S" else "p\\.\\s?s\\.", body.lower()
            )
        ):
            ps = self._postscript(prompt, body, p["postscript"], log)
        if p["constrained"] and (not any((o in body for o in OPTIONS))):
            low = body.lower()
            opt = (
                OPTIONS[1]
                if re.search("\\bno\\b", low) and (not re.search("\\byes\\b", low))
                else OPTIONS[0]
                if re.search("\\byes\\b", low)
                else OPTIONS[2]
            )
        extra = sum((count_words(x) for x in (title, ps, opt, p["end_phrase"] or "")))
        if p["max_words"] and count_words(body) + extra >= p["max_words"]:
            body = truncate_words(body, max(1, p["max_words"] - extra))
            log.append(f"truncated to {count_words(body)} words")
            body2, dropped = drop_dangling_intro(body, hl)
            if dropped:
                self.g4("dangling intro line removed after truncation", log, "H:dangling_colon")
                body = body2
        if (
            p["max_sent"]
            and count_sent(body) + (1 if ps else 0) + (1 if opt else 0) >= p["max_sent"]
        ):
            body = truncate_sentences(
                body, max(1, p["max_sent"] - (1 if ps else 0) - (1 if opt else 0))
            )
            log.append("truncated sentences")
        if (
            p["placeholders"]
            and sum((1 for a, e, inner in ph_official(body) if ph_valid4(body, a, e, inner, eng)))
            < p["placeholders"]
            and ("number_placeholders" not in self.unmet)
        ):
            self.unmet.append("number_placeholders")
        hl_lits = []
        if p["highlights"]:
            body2, nmeta = strip_meta_highlights(body, hl)
            if nmeta:
                self.g4(f"{nmeta} meta-label highlight(s) removed", log, "E:hl_meta")
                body = body2
            have = valid_highlight_count(body, hl, eng)
            if have < p["highlights"]:
                body, hl_lits = add_highlights(
                    body,
                    p["highlights"] - have,
                    start_at=len(prefix) if prefix and body.startswith(prefix) else 0,
                )
                log.append(f"highlights +{len(hl_lits)}")
                if have + len(hl_lits) < p["highlights"]:
                    self.unmet.append("number_highlighted_sections")
                    log.append("highlights unmet")
        if p["blocks"] and p["blocks"]["kind"] == "paragraphs":
            body = re.sub("\\n\\*\\*\\*(?:[ \\t]*\\n[ \\t]*\\*\\*\\*)+[ \\t]*\\n", "\n***\n", body)
        text = body
        if prefix and (not text.startswith(prefix)):
            text = prefix + text.lstrip()
            log.append("prefix restored")
            self.weak.append("prefix_restored")
        head = prefix if prefix and text.startswith(prefix) else ""
        ins_at = 1 if head else 0
        if opt:
            text = head + opt + "\n\n" + text[len(head) :]
            lits.insert(ins_at, opt)
        if title:
            tail_text = text[len(head) :] if head else text
            first_line, _, rest = tail_text.partition("\n")
            fl = first_line.strip()
            if (fl.lower().startswith("title:") and len(fl) < 120) and rest.strip():
                text = head + rest.lstrip()
            if head and (not text.startswith(head)):
                head = ""
            text = head + f"<<{title}>>\n\n" + text[len(head) :]
            lits.insert(ins_at, f"<<{title}>>")
        lits.extend(hl_lits)
        if ps:
            text = text.rstrip() + "\n\n" + ps
            lits.append(ps)
        if p["end_phrase"]:
            if "H:end_meta" in g4_issues(
                text, dict(end_phrase=p["end_phrase"], prefix=head, language=p.get("language"))
            ):
                t2, ok = fix_end_meta(text, p["end_phrase"], len(head))
                if ok:
                    self.g4(
                        "end phrase was quoted inside a meta-statement -> statement dropped, phrase appended",
                        log,
                        "H:end_meta",
                    )
                    text = t2
        if p["end_phrase"] and (
            not text.rstrip().rstrip('"').lower().endswith(p["end_phrase"].lower())
        ):
            if not re.search("[.!?\"\\')\\]]\\s*$", text):
                cut = sentence_end_before(text, len(text))
                if cut >= 0.6 * len(text) and cut > len(head):
                    text = text[:cut].rstrip()
                    log.append("trimmed to sentence end before end phrase")
            joiner = " " if re.search("[.!?\"\\')\\]]\\s*$", text) else "\n\n"
            text = text.rstrip() + joiner + p["end_phrase"]
            lits.append(p["end_phrase"])
        tf = []
        if p["no_comma"]:
            tf.append(lambda x: x.replace(",", ""))
        if p["case"] == "upper":
            tf.append(str.upper)
        if p["case"] == "lower":
            tf.append(str.lower)
        for f in tf:
            text = f(text)
            lits = [f(x) for x in lits]
        if p["quotation"]:
            text = '"' + text.strip().strip('"').strip() + '"'
            lits = ['"'] + lits + ['"']
        spans, pos = ([], 0)
        for lit in lits:
            if not lit:
                continue
            j = text.find(lit, pos)
            if j >= 0:
                spans.append((j, j + len(lit)))
                pos = j + len(lit)
        tail_txt = text[len(prefix) :] if prefix and text.startswith(prefix) else text
        final_tags = sorted(
            set(degenerate_reasons(tail_txt))
            | filler_tags(tail_txt, slots=plan_slots(p)) - {"meta_note", "echo_only"}
        )
        if final_tags:
            log.append(f"final detectors: {final_tags}")
            self.weak.append("final:" + ",".join(final_tags))
        g4f = g4_issues(text, p)
        if g4f:
            log.append(f"g4: final tags {sorted(g4f)}")
            for tg in sorted(g4f):
                if "g4:final:" + tg not in self.weak:
                    self.weak.append("g4:final:" + tg)
                c = G4_CONSTRAINT.get(tg)
                if c == "blocks" and p["blocks"]:
                    c = BLOCK_UNMET[p["blocks"]["kind"]]
                if c and c != "blocks" and (c not in self.unmet):
                    self.unmet.append(c)
        return dict(
            text=text,
            plan=p,
            clamp_spans=spans,
            gen_seconds=self.cost,
            log=log,
            unmet=sorted(set(self.unmet)),
            weak=self.weak,
            regens=self.regens,
            seconds=time.perf_counter() - t0,
            program="gen4",
            g4_final=sorted(g4f),
            gen_calls=self.ncalls,
        )

    def guard(self, orig, cert, plan_=None):
        """Reject damaging certification edits relative to the constructed text. Check any guard tag the
        certified text has and the program text has not (A-I), English non-words introduced by the MAP (mangled /
        misspelled / split words), and new broken sentence starts ('Tripoli. was')."""
        kws = tuple(plan_.get("keywords", ())) if plan_ else ()
        fbs = tuple(plan_.get("forbidden", ())) if plan_ else ()
        oc = Counter(WORD_RE.findall(orig))
        new = Counter(WORD_RE.findall(cert)) - oc
        tags = set(splice_tags(cert, new, kws, fbs))
        sl = plan_slots(plan_) if plan_ else None
        tags |= {
            "junk_" + t
            for t in filler_tags(cert, slots=sl) - filler_tags(orig, slots=sl)
            if t not in ("meta_note", "echo_only")
        }
        if CYR_GREEK.search(cert) and (not CYR_GREEK.search(orig)):
            tags.add("script_introduced")
        bad_words = sorted((w for w in new if splice_tags(w, [w], kws, fbs)))
        if G4_NO_GUARDS:
            return (sorted(tags), bad_words)
        pl = plan_ or {}
        g_c, g_o = (g4_issues(cert, pl), g4_issues(orig, pl))
        tags |= {"g4:" + t for t in set(g_c) - set(g_o)}
        if is_english(cert, pl):
            letters = [c for c in cert if c.isalpha() and c.isascii()]
            caps = letters and sum((c.isupper() for c in letters)) / len(letters) > 0.7
            kwl = {k.lower() for k in kws}
            nw = sorted(
                (
                    w
                    for w in new
                    if w not in oc
                    and w.isalpha()
                    and (len(w) >= 3)
                    and (w.lower() not in kwl)
                    and (w.islower() or caps)
                    and (not is_word(w))
                )
            )
            if nw:
                tags.add("g4:G:new_nonword")
                bad_words = sorted(set(bad_words) | set(nw))
            if (
                not pl.get("case")
                and len(re.findall("[.!?] [a-z]", cert)) - len(re.findall("[.!?] [a-z]", orig)) >= 2
            ):
                tags.add("g4:I:broken_sentences")
        ow = {w for w in oc if len(w) >= 4 and w.isalpha()}
        sp = sorted(
            (
                w
                for w in new
                if w not in oc
                and w.isalpha()
                and (len(w) >= 2)
                and (w.lower() not in {k.lower() for k in kws})
                and (not (len(w) > 2 and is_word(w)))
                and any((len(o) > len(w) and (o.startswith(w) or o.endswith(w)) for o in ow))
            )
        )
        if sp:
            tags.add("g4:G:split_word")
            bad_words = sorted(set(bad_words) | set(sp))
        st = (
            len(pl.get("prefix") or "") if pl.get("prefix") and cert.startswith(pl["prefix"]) else 0
        )
        so = (
            len(pl.get("prefix") or "") if pl.get("prefix") and orig.startswith(pl["prefix"]) else 0
        )
        eng = is_english(cert[st:], pl)
        for kr in pl.get("kw_req", ()):
            kw = kr["kw"]
            if (
                len(re.findall(re.escape(kw), cert, re.I))
                > len(re.findall(re.escape(kw), orig, re.I))
                and genuine_kw_count(cert, kw, st, eng)[0] <= genuine_kw_count(orig, kw, so, eng)[0]
            ):
                tags.add("g4:G:kw_inserted_fake")
        if pl.get("placeholders"):
            vc = sum(
                (1 for a, e, i in ph_official(cert) if a >= st and ph_valid4(cert, a, e, i, eng))
            )
            vo = sum(
                (1 for a, e, i in ph_official(orig) if a >= so and ph_valid4(orig, a, e, i, eng))
            )
            if len(ph_official(cert)) > len(ph_official(orig)) and vc <= vo:
                tags.add("g4:D:ph_inserted_fake")
        if pl.get("highlights"):
            if len(hl_official(cert)) > len(hl_official(orig)) and valid_highlight_count(
                cert, st, eng
            ) <= valid_highlight_count(orig, so, eng):
                tags.add("g4:E:hl_inserted_fake")
        if (
            pl.get("min_sent")
            and loose_sentence_count(cert) > loose_sentence_count(orig)
            and (genuine_sentence_count(cert) <= genuine_sentence_count(orig))
        ):
            tags.add("g4:F:sent_inserted_fake")
        mo, mc = (markup_profile(orig, pl), markup_profile(cert, pl))
        if mc[0] > mo[0] or mc[1] > mo[1]:
            tags.add("g4:A:stray_marker")
        if (
            "�" in cert
            and "�" not in orig
            or len(CTRL_JUNK.findall(cert)) > len(CTRL_JUNK.findall(orig))
        ):
            tags.add("g4:I:bad_chars")
        return (sorted(tags), bad_words)

    def certify_text(
        self, prompt_entry, text, clamp_spans, event, tail=160, plan=None, max_resolves=2
    ):
        torch = self.T.torch
        eos = getattr(self.T, "eos_id", 2)
        pad = getattr(self.T, "pad_id", self.tok.pad_token_id)
        ids = self.enc(text) + [eos]
        A = getattr(self.T, "canvas_total", 2048) - len(prompt_entry["prefix_ids"])
        if len(ids) > A - 8:
            ids = ids[: A - 8]
        answer = ids + [pad] * (A - len(ids))
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        clamp = [False] * A
        for a, b in clamp_spans:
            for j, (s, e) in enumerate(enc["offset_mapping"]):
                if j < len(ids) and s < b and (e > a):
                    clamp[j] = True
        free = [not c for c in clamp]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = self.T.harmonize_batch(
            prompt_entry, answer, [free], tstar=self.re_t, seeds=[self.seed + 999]
        )[0]
        torch.cuda.synchronize()
        t_re = time.perf_counter() - t0
        N = min(A, len(ids) + tail)
        x = logits[:N].double().numpy()
        x[: len(ids) - 1, eos] = -10000.0
        x[: len(ids) - 1, pad] = -10000.0
        t1 = time.perf_counter()
        reasons, resolves, rejected_text = ([], 0, None)
        while True:
            out = certify(x, event, fixed_ids=answer, fixed_mask=clamp[:N], truncate=None)
            new = list(out) + answer[N:]
            cert_text = self.T.decode(new)[0]
            reasons, bad = self.guard(text, cert_text, plan)
            if not reasons:
                break
            rejected_text = cert_text
            if resolves >= max_resolves:
                break
            hit = 0
            for j in range(min(N, len(new))):
                if new[j] == answer[j] or clamp[j]:
                    continue
                piece = self.T.decode([new[j]])[0].strip().lower()
                if len(piece) >= 2 and any((piece in w.lower() or w.lower() in piece for w in bad)):
                    x[j, new[j]] = -10000.0
                    hit += 1
            if not hit:
                for j in range(min(N, len(new))):
                    if new[j] != answer[j] and (not clamp[j]):
                        x[j, new[j]] = -10000.0
                        hit += 1
            resolves += 1
        if reasons:
            new = answer
            cert_text = text
        return dict(
            ids=new,
            text=cert_text,
            re_seconds=t_re,
            map_seconds=time.perf_counter() - t1,
            edits=sum((a != b for a, b in zip(new, answer))),
            cert_rejected=bool(reasons),
            reasons=reasons,
            resolves=resolves,
            rejected_text=rejected_text,
        )


if G4_NO_GUARDS:

    def g4_issues(text, plan_, prefix=None):
        return {}

    def g4_degenerate(text, plan_=None):
        return []

    def g4_piece_issues(cand, b, i, plan_):
        return []

    def g4_clean_block(cand, b):
        return cand

    def drop_dangling_intro(body, start=0):
        return (body, 0)

    def drop_recycled(cont, body):
        return (cont, 0)

    def strip_plan_markers(cont, p):
        return cont

    def strip_meta_highlights(body, start=0):
        return (body, 0)

    def hl_valid4(text, a, b, inner, english=True):
        return valid_highlight(inner)

    def ph_valid4(text, a, b, inner, english=True):
        return valid_placeholder(inner)

    def title_valid4(text, a, b, inner, english=True):
        return valid_title(inner)

    def title_candidate_ok(cand, english=True):
        return (
            valid_title(cand)
            and 2 <= len(cand.split()) <= 12
            and (len(ALPHA_RE.findall(cand)) >= 2)
            and (not splice_tags(cand))
        )

    G4_NATIVE_BLOCK_VETO = frozenset()

import pace.constraints.ifbench.registry as PI
from pace.constraints.ifbench.registry import (
    REGISTRY,
    POSITIVES,
    NOTES,
    PROGRAM_IDS,
    UNSUPPORTED_IDS,
)
from pace.constraints.base import DFA, DEAD, Unsupported, build, nonempty, product, minimize
from pace.constraints.ifeval import no_homoglyph, no_char_runs, no_classes, GENERIC_IDS
from pace.automata.alphabet import is_space_class, is_word_class

TEMPLATE_MIN_WORDS = 3
LAST_SENTENCE_MIN_WORDS = 4
QUOTE_DEPTH = 24
PAREN_DEPTH = 5
GENERIC_CHAR_RUNS = False
HOMOGLYPH_SKIP_IF = ("count:words_japanese",)
TEMPLATE_MARKERS = ("My Answer:", "My Conclusion:", "Future Outlook:")


def nested_quotes_balanced(depth_cap=QUOTE_DEPTH):
    """Mirror of NestedQuotesChecker (quote equal to the stack top pops, any other quote pushes; True once a pop
    brings depth to reached-3); accept iff that happened AND the stack is empty at the end. Class 132 (curly quotes,
    other symbols) is dead; an apostrophe between two word characters is dead."""

    def top(bottom, depth):
        return bottom if depth % 2 == 1 else 34 if bottom == 39 else 39

    def step(s, c):
        bottom, depth, reached, acc, prev_word, pend_apos = s
        if c == 132:
            return DEAD
        wc = is_word_class(c)
        if pend_apos and wc:
            return DEAD
        pend = False
        if depth and c == top(bottom, depth):
            depth -= 1
            if reached - depth >= 3:
                acc = True
            if depth == 0:
                bottom = 0
            pend = c == 39 and prev_word
        elif c in (34, 39):
            if depth >= depth_cap:
                return DEAD
            if depth == 0:
                bottom = c
            depth += 1
            reached = max(reached, depth)
            pend = c == 39 and prev_word
        return (bottom, depth, reached, acc, wc, pend)

    return build((0, 0, 0, False, False, False), step, lambda s: s[3] and s[1] == 0)


def nested_parentheses_balanced(depth_cap=PAREN_DEPTH):
    """Mirror of NestedParenthesesChecker; a mismatched close or a close on an empty stack is dead (the official
    checker resets); accept iff the official condition held AND the stack is empty at the end."""
    OPEN = {40: 41, 91: 93, 123: 125}
    CLOSE = {41, 93, 125}

    def step(s, c):
        stack, reached, acc = s
        if c in OPEN:
            if len(stack) >= depth_cap:
                return DEAD
            stack = stack + (OPEN[c],)
            return (stack, max(reached, len(stack)), acc)
        if c in CLOSE:
            if not stack or stack[-1] != c:
                return DEAD
            stack = stack[:-1]
            return (stack, reached, acc or (reached >= 5 and len(stack) < reached))
        return s

    return build(((), 0, False), step, lambda s: s[2] and len(s[0]) == 0)


def _aho(patterns):
    """Deterministic Aho-Corasick over byte-code tuples: returns (goto dict list, output list of pattern ids)."""
    nodes = [{}]
    out = [set()]
    fail = [0]
    for pid, p in enumerate(patterns):
        n = 0
        for ch in p:
            if ch not in nodes[n]:
                nodes[n][ch] = len(nodes)
                nodes.append({})
                out.append(set())
                fail.append(0)
            n = nodes[n][ch]
        out[n].add(pid)
    order = []
    queue = [v for v in nodes[0].values()]
    for v in queue:
        fail[v] = 0
    while queue:
        u = queue.pop(0)
        order.append(u)
        for ch, v in nodes[u].items():
            f = fail[u]
            while f and ch not in nodes[f]:
                f = fail[f]
            fail[v] = nodes[f][ch] if ch in nodes[f] and nodes[f][ch] != v else 0
            out[v] |= out[fail[v]]
            queue.append(v)
    alphabet = set((ch for p in patterns for ch in p))

    def goto(n, ch):
        while n and ch not in nodes[n]:
            n = fail[n]
        return nodes[n].get(ch, 0)

    table = {(n, ch): goto(n, ch) for n in range(len(nodes)) for ch in alphabet}
    return (table, out, alphabet)


def template_fields(min_words=TEMPLATE_MIN_WORDS, markers=TEMPLATE_MARKERS):
    """After each marker the field text must hold >= min_words word runs before the next marker / the end. Words are
    counted at run end, so the marker's own two words are already in the count when it completes (subtract 2)."""
    pats = [tuple((ord(ch) for ch in m)) for m in markers]
    table, out, alphabet = _aho(pats)
    cap = min_words + 2

    def step(s, c):
        node, open_, count, in_word = s
        wc = is_word_class(c)
        if in_word and (not wc):
            count = min(cap, count + 1)
        node = table.get((node, c), 0) if c in alphabet else 0
        if out[node]:
            if open_ and count < cap:
                return DEAD
            return (node, True, 0, False)
        return (node, open_, count, wc)

    def accept(s):
        node, open_, count, in_word = s
        return not open_ or min(cap, count + int(in_word)) >= min_words

    return build((0, False, 0, False), step, accept)


def last_sentence_words(min_words=LAST_SENTENCE_MIN_WORDS):
    """Every non-empty '\\n'-line: the text after its last sentence boundary ([.!?] followed by whitespace) holds
    >= min_words word runs (a line with a single token is therefore dead)."""
    cap = min_words
    END = frozenset(map(ord, ".!?"))

    def step(s, c):
        words, in_word, pend, has_tok = s
        if c == 10:
            if has_tok and words < cap:
                return DEAD
            return (0, False, False, False)
        if is_space_class(c):
            return (0 if pend else words, False, False, has_tok)
        wc = is_word_class(c)
        if wc and (not in_word):
            words = min(cap, words + 1)
        return (words, wc, c in END, True)

    return build((0, False, False, False), step, lambda s: not s[3] or s[0] >= cap)


def interrobang_context():
    """Every '?!' / '!?' bigram must directly follow a word character and be followed by whitespace or the end
    (kills '(!?!?)', '(!?:)', 'all?!)?;', 'its?!.', '??!!Intro' clusters). Extra condition over the official DFA."""

    def kind(c):
        if c == 63:
            return "q"
        if c == 33:
            return "e"
        return "w" if is_word_class(c) else "o"

    def step(s, c):
        prev2_word, prev, need_ws = s
        if need_ws and (not is_space_class(c)):
            return DEAD
        k = kind(c)
        need = False
        if k in ("q", "e") and prev in ("q", "e") and (prev != k):
            if not prev2_word:
                return DEAD
            need = True
        return (prev == "w", k, need)

    return build((False, "o", False), step, lambda s: True)


REPLACED = {
    "format:quotes": lambda kw, prompt: [nested_quotes_balanced()],
    "format:parentheses": lambda kw, prompt: [nested_parentheses_balanced()],
}
EXTRA = {
    "count:keywords_multiple": lambda kw, prompt: [no_classes((128, 129, 130, 131))],
    "format:output_template": lambda kw, prompt: [template_fields()],
    "words:paragraph_last_first": lambda kw, prompt: [last_sentence_words()],
    "count:punctuation": lambda kw, prompt: [interrobang_context()],
}
HARDENED_IDS = tuple(sorted(set(REPLACED) | set(EXTRA)))


def predicate(iid, kw, prompt=None):
    kw = {k: v for k, v in (kw or {}).items() if v is not None}
    if iid.startswith("custom:"):
        raise Unsupported("custom:* checks a task-specific answer (not compiled)")
    if iid in REPLACED:
        return REPLACED[iid](kw, prompt)
    base = PI.predicate(iid, kw, prompt)
    if iid in EXTRA:
        base = list(base) + EXTRA[iid](kw, prompt)
    return base


def generic_components(row):
    comps = []
    if not any((i in row["instruction_id_list"] for i in HOMOGLYPH_SKIP_IF)):
        comps.append((-1, "generic:no_homoglyph", no_homoglyph()))
    if GENERIC_CHAR_RUNS:
        comps.append((-1, "generic:no_char_runs", no_char_runs()))
    return comps


def _components(row, generic):
    comps, unsupported = ([], [])
    for idx, (iid, kw) in enumerate(zip(row["instruction_id_list"], row["kwargs"])):
        if iid in PROGRAM_IDS:
            unsupported.append((idx, iid, "program-handled"))
            continue
        try:
            for d in predicate(iid, kw, row["prompt"]):
                comps.append((idx, iid, d))
        except Unsupported as e:
            unsupported.append((idx, iid, str(e)))
    if generic:
        comps.extend(generic_components(row))
    return (comps, unsupported)


def _fits(comps, limit):
    try:
        product([nonempty()] + [d for _, _, d in comps], max_states=limit)
        return True
    except ValueError:
        return False


def components_for(row, generic=True, max_text_states=None, drop=False):
    """Hardened components (+ generic). With max_text_states the joint text product is kept under the budget by degrading
    the most expensive HARDENED instruction (types in HARDENED_IDS) back to its official predicates_ifb DFAs, one step
    at a time (generic:no_char_runs is dropped first); steps are reported in `unsupported` as 'degraded:...'."""
    comps, unsupported = _components(row, generic)
    if max_text_states is None or _fits(comps, max_text_states):
        return (comps, unsupported)
    if any((c[1] == "generic:no_char_runs" for c in comps)):
        comps = [c for c in comps if c[1] != "generic:no_char_runs"]
        unsupported.append((-1, "generic:no_char_runs", "degraded:dropped (text budget)"))
    done = set()
    while not _fits(comps, max_text_states):
        cost = {}
        for i, iid, d in comps:
            if i >= 0 and iid in HARDENED_IDS and (i not in done):
                cost[i] = cost.get(i, 0) + len(d.states)
        if not cost:
            break
        idx = max(cost, key=cost.get)
        done.add(idx)
        iid, kw = (row["instruction_id_list"][idx], row["kwargs"][idx])
        comps = [c for c in comps if c[0] != idx]
        try:
            official = PI.predicate(iid, kw, row["prompt"])
        except Unsupported:
            official = []
        comps.extend(((idx, iid, d) for d in official))
        unsupported.append((idx, iid, "degraded:official-only (text budget)"))
    while drop and (not _fits(comps, max_text_states)):
        size = {}
        for i, iid, d in comps:
            if i >= 0:
                size[i] = size.get(i, 1) * max(1, len(d.states))
        if not size:
            break
        idx = max(size, key=size.get)
        comps = [c for c in comps if c[0] != idx]
        unsupported.append(
            (idx, row["instruction_id_list"][idx], "dropped:program-handled (state budget)")
        )
    return (comps, unsupported)


DEFAULT_LADDER = (None, 5000, 2000, 1000, 0)


def components_ladder(row, budgets=DEFAULT_LADDER, generic=True):
    """Yield (budget, comps, unsupported) for successively smaller text budgets. The token-graph size is only loosely
    predicted by the text product (block / word-count DFAs expand 30-140x per token context), so a compiler should
    try each level until compile_contextual_event fits its budget; budget 0 = official-only components
    (every hardened instruction degraded, generic:no_homoglyph kept)."""
    seen = set()
    for b in budgets:
        comps, unsupported = components_for(row, generic=generic, max_text_states=b)
        sig = tuple(((i, iid, len(d.states)) for i, iid, d in comps))
        if sig in seen:
            continue
        seen.add(sig)
        yield (b, comps, unsupported)
    yield ("nogeneric", *components_for(row, generic=False, max_text_states=0))
    for b in (5000, 1000):
        yield (f"drop{b}", *components_for(row, generic=False, max_text_states=b, drop=True))

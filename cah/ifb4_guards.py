import difflib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'cah'))
import program_gen4 as G4
from program_gen4 import WORD_RE, ALPHA_RE, norm, units, repeated_spans, recycled_units, is_word, stop_en
TOK_RE = re.compile("[^\\W_]+(?:['’][^\\W_]+)*")
COUNT_ITEM_IIDS = ('count:keywords_multiple', 'count:person_names', 'words:palindrome', 'count:unique_word_count', 'sentence:keyword', 'count:pronouns', 'count:numbers', 'count:conjunctions', 'format:emoji', 'count:punctuation', 'words:words_position', 'words:keywords_specific_position')
LIST_FORMAT_IIDS = ('format:list', 'format:sub-bullets', 'format:no_bullets_bullets', 'format:output_template', 'count:numbers', 'count:word_count_range')
CASE_IIDS = ('repeat:repeat_simple', 'repeat:repeat_change', 'repeat:repeat_span', 'format:title_case')
LANG = {'count:words_japanese': 'ja'}
NO_LEXICAL = ('format:no_whitespace',)
OTHER_SCRIPT = re.compile('[Ѐ-ӿ\u3040-ヿ一-鿿가-\ud7af\u0e00-\u0e7f\u0600-ۿ]')
META_TAIL = re.compile('(?i)\\b(?:this|the|my|our) (?:answer|response|reply|text|essay|list|output|story)\\s+(?:is|has|was|ends|will)\\b|\\bthat is all\\b|\\bnothing (?:more|further|else)\\b|\\b(?:words?|terms?|names?|vocabulary|palindromes?|keywords?) (?:worth noting|to (?:note|remember)|used|included)\\b')
CUT_END = re.compile('(?i)\\b(?:the|a|an|of|to|for|with|from|by|and|or|but|than|its|their|our|your|my|into|onto|upon)\\s*[.!?]+[\\"\')\\]]*\\s*$')
UNIT_SUFFIX = {'th', 'st', 'nd', 'rd', 'km', 'kg', 'mg', 'mm', 'cm', 'ml', 'am', 'pm', 'gb', 'mb', 'kb', 'hz', 'px', 'ms', 'mph', 'kph', 'kwh', 'bc', 'ad', 'bce', 'ce', 'yo', 'lb', 'lbs', 'oz', 'ft', 'in', 'x', 'k', 's', 'gen', 'mins', 'min', 'sec', 'secs', 'hrs', 'hr', 'yrs', 'mbps', 'gbps', 'fps', 'dpi', 'ppm', 'mah'}
_STOP = None

def stops():
    global _STOP
    if _STOP is None:
        _STOP = frozenset(stop_en()) | frozenset(G4.STOP) | frozenset(("n't", "'s", 'not'))
    return _STOP

def toks(text):
    return [m.group().lower().replace('’', "'") for m in TOK_RE.finditer(text)]

def is_content(t):
    """Content token: any token with a digit, or an alphabetic token of >= 3 letters that is not a stop word."""
    if any((c.isdigit() for c in t)):
        return True
    return len(t) >= 3 and t not in stops()

def lang_of(row):
    for iid in row['instruction_id_list']:
        if iid in LANG:
            return LANG[iid]
    return None

def kw_literals(row):
    """Every string the instruction kwargs mandate (keywords, word, sep, options, request to repeat, ...)."""
    out = []
    for kw in row['kwargs']:
        for k, v in (kw or {}).items():
            if v is None:
                continue
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, (list, tuple)):
                out += [x for x in v if isinstance(x, str)]
    return out

def literal_tokens(row):
    s = set()
    for x in kw_literals(row):
        s |= set(toks(x))
    return s

def _remove_spans(text, spans):
    out, pos = ([], 0)
    for a, b in sorted(spans):
        if a < pos:
            a = pos
        out.append(text[pos:a])
        pos = max(pos, b)
    out.append(text[pos:])
    t = ''.join(out)
    t = re.sub('[ \\t]{2,}', ' ', t)
    t = re.sub(' +\\n', '\n', t)
    t = re.sub('\\n{3,}', '\n\n', t)
    return t.strip()
LABEL_RE = re.compile('^\\W{0,3}([^\\W\\d_][^:\\n]{0,48}?)\\s*:\\s+(\\S.*)$')
LIST_LINE = re.compile('^\\s*(?:[-*•+]|\\d{1,3}[.)])\\s+([^\\s,;:]+(?:\\s+[^\\s,;:]+)?)\\s*[.,;]?\\s*$')

def dump_units(text):
    """Appendix-list units [(start, end, kind)]: hashtag runs, '<label>: a, b(, c)' lines (>= 2 items of <= 3 words),
    bare comma lists (>= 4 items of <= 2 words, nothing else on the line), runs of >= 3 one/two-word list lines."""
    out = []
    lines = list(re.finditer('[^\\n]+', text))
    for m in lines:
        s = m.group().strip()
        if not s:
            continue
        hs = re.findall('#[^\\W_][\\w]*', s)
        if len(hs) >= 2 and len(WORD_RE.findall(re.sub('#[^\\W_][\\w]*', ' ', s))) <= len(hs):
            out.append((m.start(), m.end(), 'hashtags'))
            continue
        body, labelled = (s, False)
        lm = LABEL_RE.match(s)
        if lm and len(WORD_RE.findall(lm.group(1))) <= 6:
            body, labelled = (lm.group(2), True)
        body = re.sub('^(?:[-*•+]|\\d{1,3}[.)])\\s+', '', body).strip().rstrip(' .!;')
        if re.search('[.!?]\\s', body):
            continue
        items = [x.strip() for x in re.split('\\s*[,;、]\\s*', body) if x.strip()]
        items = [re.sub('^(?:and|or)\\s+', '', x) for x in items]
        if labelled and len(items) >= 2 and all((1 <= len(WORD_RE.findall(x)) <= 3 for x in items)):
            out.append((m.start(), m.end(), 'label_list'))
            continue
        if not labelled and len(items) >= 4 and all((1 <= len(WORD_RE.findall(x)) <= 2 for x in items)):
            out.append((m.start(), m.end(), 'bare_list'))
    run = []
    for m in lines + [None]:
        if m is not None and LIST_LINE.match(m.group()):
            run.append(m)
            continue
        if len(run) >= 3:
            out += [(x.start(), x.end(), 'list_lines') for x in run]
        run = []
    for m in re.finditer('(?:#[^\\W_][\\w]*[\\s,.;]*){2,}', text):
        if not any((a <= m.start() < b for a, b, _ in out)):
            out.append((m.start(), m.end(), 'hashtags'))
    out += template_items(text)
    return sorted(set(out))
LIST_ITEM_LINE = re.compile('^\\s*(?:[-*+]|\\d{1,3}[.)])\\s+(\\S.*)$')

def template_items(text, min_items=3, jac=0.6):
    """Runs of >= 3 list items that are near-identical apart from one or two words ('1. Emma in a forest with meadow',
    '2. Liam in a forest with meadow', ...): template-filled lists."""
    out, run = ([], [])
    lines = list(re.finditer('[^\\n]+', text))

    def flush():
        if len(run) >= min_items:
            sets = [set(toks(LIST_ITEM_LINE.match(m.group()).group(1))) for m in run]
            sims = [len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])) for i in range(len(sets)) for j in range(i)]
            if sims and sum(sims) / len(sims) >= jac and (min((len(x) for x in sets)) >= 3):
                out.extend(((m.start(), m.end(), 'template_items') for m in run))
    for m in lines:
        if LIST_ITEM_LINE.match(m.group()):
            run.append(m)
            continue
        if m.group().strip():
            flush()
            run = []
    flush()
    return out

def content_prefixes(text):
    return {t[:5] for t in toks(text) if t.isalpha() and len(t) >= 4 and (t not in stops())}

def junk_segment(seg, rest, kind='last_line'):
    """Why a tail segment is junk (None if it is plausible content). 'detached' (no shared content word with the rest of
    the answer) is only judged for a separate last line (canned sentences are appended as their own line)."""
    w = WORD_RE.findall(seg)
    if not w:
        return None
    if META_TAIL.search(seg):
        return 'meta'
    cw = [t for t in toks(seg) if t.isalpha() and len(t) >= 4 and (t not in stops())]
    if kind == 'last_line' and len(cw) >= 3 and (len(WORD_RE.findall(rest)) >= 20) and (not any((t[:5] in content_prefixes(rest) for t in cw))):
        return 'detached'
    g = G4._grams(seg, 3)
    if len(g) >= 2 and len(g & G4._grams(rest, 3)) / len(g) >= 0.6:
        return 'recycled'
    if len(w) >= 3:
        rn = norm(rest)
        sn = norm(seg)
        if len(sn.split()) >= 3 and sn in rn:
            return 'recycled'
    return None

def tail_segments(text):
    """Candidate junk tails: [(start, end, kind)] -- the last line, its last sentence, a trailing dash clause; plus the
    group of <= 2-word final sentences of the lines (anchors appended to every paragraph)."""
    lines = [(m.start(), m.end()) for m in re.finditer('[^\\n]*\\S[^\\n]*', text)]
    if not lines:
        return ([], [])
    cands = []
    ls, le = lines[-1]
    if len(lines) >= 2:
        cands.append((ls, le, 'last_line'))
    us = units(text[ls:le], 0)
    if len(us) >= 2:
        a, b, _ = us[-1]
        cands.append((ls + a, ls + b, 'last_sentence'))
    line = text[ls:le]
    dm = list(re.finditer('\\s[—–]\\s|\\s--\\s|,\\s+(?:that is|i\\.e\\.|namely)\\b', line))
    if dm:
        p = dm[-1].start()
        if 1 <= len(WORD_RE.findall(line[p:])) <= 25 and len(WORD_RE.findall(line[:p])) >= 3:
            cands.append((ls + p, le, 'dash_clause'))
    anchors = []
    for s, e in lines:
        us = units(text[s:e], 0)
        if len(us) >= 2:
            a, b, sent = us[-1]
            sw = [x.lower() for x in WORD_RE.findall(sent)]
            earlier = {x.lower() for x in WORD_RE.findall(text[s:s + a])}
            if 1 <= len(sw) <= 2 and all((x in earlier for x in sw)):
                anchors.append((s + a, s + b, 'anchor'))
    return (cands, anchors)

def interrobang_bad(text, literals=()):
    t = text
    for lit in literals:
        if lit and ('?!' in lit or '!?' in lit):
            t = t.replace(lit, ' ')
    bad = 0
    for m in re.finditer('\\?!|!\\?', t):
        a, b = (m.start(), m.end())
        before_ok = a > 0 and t[a - 1].isalnum()
        after_ok = b >= len(t) or t[b].isspace() or t[b] in '"\')”’'
        if not (before_ok and after_ok):
            bad += 1
    return bad

def glued_digits(text, prompt=''):
    """Digits glued to an English word ('serial1killer', 'Novel0', '1On'); units / ordinals / prompt tokens excluded."""
    pt = set(toks(prompt))
    out = []
    for m in re.finditer('(?<![\\w])(?:[A-Za-z]+\\d+[A-Za-z]*|\\d+[A-Za-z]+)(?![\\w])', text):
        w = m.group()
        if w.lower() in pt:
            continue
        letters = re.findall('[A-Za-z]+', w)
        if not letters or all((l.lower() in UNIT_SUFFIX or len(l) <= 1 for l in letters)):
            continue
        if any((len(l) >= 2 and l.lower() not in UNIT_SUFFIX and is_word(l) and (len(l) >= 2) for l in letters)):
            if re.fullmatch('\\d+(?:st|nd|rd|th|s)', w, re.I):
                continue
            out.append(w)
    return out

def numbering_bad(text):
    """List numbering anomalies: a bare number marker line, or a numbered run with a duplicate / skipped number."""
    issues = []
    for m in re.finditer('(?m)^\\s*\\d{1,3}[.)]\\s*$', text):
        issues.append('bare')
        break
    if re.search('(?m)^\\s*[.)]\\s+\\S', text):
        issues.append('orphan_marker')
    nums, prev_line = ([], -2)
    lines = text.split('\n')
    runs = []
    for i, l in enumerate(lines):
        mm = re.match('^\\s*(\\d{1,3})[.)]\\s+\\S', l)
        if mm:
            if runs and i - runs[-1][-1][0] <= 2:
                runs[-1].append((i, int(mm.group(1))))
            else:
                runs.append([(i, int(mm.group(1)))])
    for r in runs:
        ns = [n for _, n in r]
        if len(ns) >= 2:
            if len(set(ns)) < len(ns):
                issues.append('duplicate')
            elif any((b != a + 1 for a, b in zip(ns, ns[1:]))):
                issues.append('skip')
    return issues
TEMPLATE_MARKERS = ('My Answer:', 'My Conclusion:', 'Future Outlook:')

def empty_template_fields(text, min_words=3):
    """Template fields (text after a marker up to the next marker / end) with fewer than min_words words."""
    pos = sorted(((m.start(), m.end()) for mk in TEMPLATE_MARKERS for m in re.finditer(re.escape(mk), text)))
    bad = 0
    for k, (a, b) in enumerate(pos):
        end = pos[k + 1][0] if k + 1 < len(pos) else len(text)
        if len(WORD_RE.findall(text[b:end])) < min_words:
            bad += 1
    return bad

def case_mangled(text):
    bad = []
    for w in re.findall('(?<![\\w])[A-Za-z]{3,}(?![\\w])', text):
        if w.islower() or w.isupper() or w.istitle():
            continue
        if re.fullmatch('[a-z][A-Z][a-z]+', w) or re.match('(?:Mc|Mac)[A-Z]', w):
            continue
        parts = re.findall('[A-Z]?[a-z]+|[A-Z]+(?![a-z])', w)
        if len(parts) >= 2 and all((len(p) >= 2 and is_word(p) for p in parts)):
            continue
        if is_word(w.lower()):
            bad.append(w)
    return bad

def ws_padding(text):
    """Whitespace / empty-field padding: a run of >= 8 blank lines or >= 12 blank lines in runs of >= 3, a non-leading
    whitespace run of >= 20 characters, >= 3 consecutive empty comma fields, or >= 4 lines made only of , . ; :"""
    runs = [len(m.group(0).split('\n')) - 2 for m in re.finditer('\\n(?:[ \\t]*\\n){3,}', text)]
    if any((r >= 8 for r in runs)) or sum(runs) >= 12:
        return 'blank-line runs'
    if any((re.search('\\S[ \\t]{20,}', l) for l in text.split('\n') if '|' not in l)):
        return 'internal whitespace run'
    if re.search(',(?:\\s*,){2,}', text):
        return 'empty comma fields'
    if len(re.findall('(?m)^[ \\t]*[,.;:][,.;: \\t]*$', text)) >= 4:
        return 'punctuation-only lines'
    return None

def script_mix(text, prompt=''):
    if OTHER_SCRIPT.search(prompt or ''):
        return None
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 40 or sum((c.isascii() for c in letters)) / len(letters) < 0.8:
        return None
    frag = [t for t in re.findall('\\S+', text) if OTHER_SCRIPT.search(t)]
    return len(frag) if len(frag) >= 2 else None

def generic_tags(text, row):
    """Tags that speak about every instruction of the row."""
    iids = row['instruction_id_list']
    lang = lang_of(row)
    tags = {}
    tail = text
    if any((i in NO_LEXICAL for i in iids)):
        for r in G4.degenerate_reasons(tail):
            tags['I:' + r] = 1
    else:
        g = G4.g4_issues(tail, dict(language=lang), prefix='')
        for k, v in g.items():
            if k.startswith('I:') or k == 'H:meta_answer':
                tags[k] = v
    if lang is None:
        sm = script_mix(text, row.get('prompt', ''))
        if sm:
            tags['I:script_mix'] = sm
    wp = ws_padding(text)
    if wp:
        tags['M:ws_padding'] = wp
    nb = text.count('�') + len(G4.CTRL_JUNK.findall(text))
    if nb and 'format:emoji' not in iids:
        tags['I:bad_chars'] = nb
    return tags

def structure_tags(text, row):
    """program_gen4 A-class / dangling colon on list-like formats (judged only for LIST_FORMAT_IIDS)."""
    g = G4.g4_issues(text, dict(language=lang_of(row)), prefix='')
    return {k: v for k, v in g.items() if k in ('A:skeleton', 'A:fragment_end', 'H:dangling_colon')}

def text_tags(text, row, official=None):
    """{idx: {tag: detail}} for each instruction idx of the row (+ -1 for the generic tags). With `official` the
    counterfactual J / K checks run for instructions the official checker passes."""
    prompt = row.get('prompt', '')
    out = {-1: generic_tags(text, row)}
    st = None
    lits = kw_literals(row)
    for idx, (iid, kw) in enumerate(zip(row['instruction_id_list'], row['kwargs'])):
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        t = {}
        ok = None
        if official is not None:
            ok = official(iid, kw, text, prompt)
        if iid == 'count:punctuation':
            n = interrobang_bad(text, lits)
            if n:
                t['L:interrobang'] = n
        if iid == 'count:numbers':
            nb = numbering_bad(text) + (['glued'] if glued_digits(text, prompt) else [])
            if nb:
                t['L:numbering'] = ','.join(sorted(set(nb)))
        if iid == 'count:word_count_range' and CUT_END.search(text.rstrip()):
            t['A:cut_end'] = 1
        if iid == 'format:output_template':
            ef = empty_template_fields(text)
            if ef:
                t['A:empty_field'] = ef
        if iid in CASE_IIDS:
            cm = case_mangled(text)
            if cm:
                t['M:case_mangled'] = len(cm)
        if iid in LIST_FORMAT_IIDS:
            if st is None:
                st = structure_tags(text, row)
            t.update(st)
        if ok:
            total = len(WORD_RE.findall(text))
            if iid in COUNT_ITEM_IIDS:
                du = dump_units(text)
                if du:
                    dw = sum((len(WORD_RE.findall(text[a:b])) for a, b, _ in du))
                    if dw <= 0.5 * total:
                        rest = _remove_spans(text, [(a, b) for a, b, _ in du])
                        if not official(iid, kw, rest, prompt):
                            t['J:dumped_items'] = ','.join(sorted({k for _, _, k in du}))
            cands, anchors = tail_segments(text)
            groups = [[c] for c in cands] + ([anchors] if anchors else [])
            for grp in groups:
                spans = [(a, b) for a, b, _ in grp]
                rest = _remove_spans(text, spans)
                if not rest or len(WORD_RE.findall(rest)) < 3:
                    continue
                if grp is anchors:
                    why = 'anchor'
                else:
                    a, b, kind = grp[0]
                    why = junk_segment(text[a:b], rest, kind)
                if why and (not official(iid, kw, rest, prompt)):
                    t['K:junk_tail'] = f'{grp[0][2]}:{why}'
                    break
        if t:
            out[idx] = t
    return out

def relevant(tags_by_idx, idx):
    """Tags that bear on instruction idx: its own + the generic ones."""
    d = dict(tags_by_idx.get(-1, {}))
    d.update(tags_by_idx.get(idx, {}))
    return d

def _unit_set(text):
    return {norm(s) for _, _, s in units(text, 1)}

def _is_glue(t, pieces, depth=0):
    """t is a concatenation of 2-3 tokens of `pieces` (a token the transform formed by deleting a separator, e.g. '*')."""
    for i in range(1, len(t)):
        if t[:i] in pieces and (t[i:] in pieces or (depth < 1 and _is_glue(t[i:], pieces, depth + 1))):
            return True
    return False

def authored_tokens(before, after, row, extra_allowed=()):
    """Content tokens the transform introduced that occur neither in the model text, the prompt nor the kwargs (tokens
    glued from model tokens by a deleted separator are not authored)."""
    bt = set(toks(before))
    allowed = bt | set(toks(row.get('prompt', ''))) | literal_tokens(row) | set(extra_allowed)
    new = Counter(toks(after)) - Counter(toks(before))
    return sorted((t for t in new if is_content(t) and t not in allowed and (not _is_glue(t, bt))))

def inserted_runs(before, after):
    """Token runs the transform inserted (difflib insert / replace ops on lower-cased tokens): [(tokens, a_start, a_end)]."""
    bt, at = (_tok_spans(before), _tok_spans(after))
    sm = difflib.SequenceMatcher(a=[x[2] for x in bt], b=[x[2] for x in at], autojunk=False)
    return [([x[2] for x in at[j1:j2]], at[j1][0], at[j2 - 1][1]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op in ('insert', 'replace') and j2 > j1]

def _tok_spans(text):
    return [(m.start(), m.end(), m.group().lower()) for m in TOK_RE.finditer(text)]

def spliced_literals(before, after, row):
    """Mandated literals the transform inserted INSIDE a model sentence: an inserted / replacing token run made of
    literal tokens whose neighbours in the model text belong to the same sentence (no sentence / line break between)."""
    lit = literal_tokens(row)
    if not lit:
        return []
    bt, at = (_tok_spans(before), _tok_spans(after))
    sm = difflib.SequenceMatcher(a=[x[2] for x in bt], b=[x[2] for x in at], autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op not in ('insert', 'replace'):
            continue
        ins = [x[2] for x in at[j1:j2]]
        if not ins or not all((x in lit for x in ins)):
            continue
        if i1 == 0 or i1 >= len(bt):
            continue
        gap_before = before[bt[i1 - 1][1]:bt[i1][0]] if op == 'insert' else before[bt[i1 - 1][1]:bt[i1][0]]
        if op == 'replace':
            if i2 >= len(bt):
                continue
            gap = before[bt[i1 - 1][1]:bt[i2][0]]
        else:
            gap = gap_before
        if not re.search('[.!?:;\\n]', gap):
            out.append(' '.join(ins))
    return out
POSITION_IIDS = ('words:words_position', 'words:keywords_specific_position')
STRICT_CANNED = os.environ.get('IFB4_STRICT_CANNED', '1') == '1'
TAG_RE = re.compile('</?[A-Za-z][A-Za-z0-9]{0,10}(?:\\s[^<>\\n]{0,60})?>')

def transform_tags(before, after, row, iid, kw=None, last=None, synonyms=()):
    """Provenance guard for one transform (before -> after). Empty dict = the edit only restructures / deletes / labels."""
    tags = {}
    if after == before:
        return tags
    if re.sub('\\s+', '', after) == re.sub('\\s+', '', before):
        return tags
    if iid.startswith('custom:') and os.environ.get('IFB4_CUSTOM_TRANSFORMS', '0') != '1':
        return {'K:prompt_specific': 'custom:* transforms are disabled (prompt-specific code)'}
    if iid.startswith('custom:') and (not STRICT_CANNED):
        ga, gb = (generic_tags(after, row), generic_tags(before, row))
        return {k: ga[k] for k in set(ga) - set(gb) if k.startswith('I:') or k == 'M:ws_padding'}
    before, after = (TAG_RE.sub(' ', before), TAG_RE.sub(' ', after))
    last = last or {}
    method, detail = (str(last.get('method', 'primary')), str(last.get('detail', '')))
    if method.startswith('fallback') or detail == 'crude-delete':
        tags['F:crude_fallback'] = method + (':' + detail if detail else '')
    if iid == 'ratio:stop_words' and detail.startswith('tier'):
        try:
            if int(detail[4:]) >= 4:
                tags['F:deep_deletion'] = detail
        except ValueError:
            pass
    au = authored_tokens(before, after, row, () if STRICT_CANNED else synonyms if iid == 'words:repeats' else ())
    if len(au) >= (1 if STRICT_CANNED else 2):
        tags['K:authored'] = ' '.join(au[:8])
    elif au:
        bu = _unit_set(before)
        for _, _, s in units(after, 1):
            if norm(s) in bu:
                continue
            cw = [t for t in toks(s) if is_content(t)]
            if cw and all((t in au or t not in set(toks(before)) for t in cw)) and any((t in au for t in cw)):
                tags['K:authored'] = ' '.join(au)
                break
    old_d = {norm(after[a:b]) for a, b, _ in dump_units(before)} if before else set()
    new_d = [after[a:b] for a, b, _ in dump_units(after) if norm(after[a:b]) not in old_d]
    if new_d:
        tags['J:dump_added'] = len(new_d)
    grown = sum((1 for t in (Counter(toks(after)) - Counter(toks(before))).elements() if is_content(t)))
    rb, ra = (len(repeated_spans(before)), len(repeated_spans(after)))
    cb, ca = (len(recycled_units(before)), len(recycled_units(after)))
    if grown >= 3 and (ra > rb or ca > cb):
        tags['F:recycled_added'] = f'repeat {rb}->{ra}, recycled {cb}->{ca}'
    added = Counter(toks(after)) - Counter(toks(before))
    btoks = set(toks(before))
    for run, a0, a1 in inserted_runs(before, after):
        cw = [t for t in run if is_content(t)]
        grownw = [t for t in cw if added[t] > 0]
        if len(run) >= 3 and len(cw) >= 2 and (len(grownw) >= 0.6 * len(cw)) and all((t in btoks for t in grownw)):
            tags['F:recycled_added'] = 'copied run: ' + ' '.join(run)[:40]
            break
        if 1 <= len(run) <= 2 and re.fullmatch('[^\\w\\n]*', after[a1:].split('\n', 1)[0]):
            ls = after.rfind('\n', 0, a0) + 1
            if all((t in toks(after[ls:a0]) for t in run)):
                tags['K:anchor_added'] = ' '.join(run)
                break
    sp = spliced_literals(before, after, row) if iid not in POSITION_IIDS else []
    if sp:
        tags['G:literal_spliced'] = '; '.join(sp[:4])
    bu = _unit_set(before)
    pn = norm(row.get('prompt', ''))
    for _, _, s in units(after, 1):
        n = norm(s)
        if not n or n in bu or n in pn:
            continue
        w = [x.lower() for x in WORD_RE.findall(s)]
        if 1 <= len(w) <= 3 and re.search('[.!?]\\s*$', s.strip()) and (not re.match('^\\s*(?:[-*•+]|\\d{1,3}[.)])', s)):
            if any((added[x] > 0 for x in w)):
                tags['K:anchor_added'] = s.strip()[:30]
                break
    ua = units(after, 0)
    if ua:
        lastu = norm(ua[-1][2])
        if lastu and lastu not in bu and any((b.startswith(lastu) and len(b) > len(lastu) + 1 for b in bu)) and CUT_END.search(after.rstrip()):
            tags['A:cut_sentence'] = ua[-1][2][-30:]
    if iid not in NO_LEXICAL and iid != 'format:newline':
        gb = generic_tags(before, row)
        ga = generic_tags(after, row)
        for k in set(ga) - set(gb):
            if k.startswith('I:') or k == 'M:ws_padding':
                tags[k] = ga[k]
    if iid == 'format:options':
        a = after.strip().strip('.').lower()
        if a and (not re.search('(?<![a-z])' + re.escape(a) + '(?![a-z])', before.lower())):
            tags['K:ungrounded_choice'] = a[:20]
    return tags

def cert_tags(orig, cert, row, official=None):
    """Tags a certification (MAP) edit introduced relative to the program text, + official regressions."""
    tags = {}
    if cert == orig:
        return tags
    prompt = row.get('prompt', '')
    to, tc = (text_tags(orig, row, official), text_tags(cert, row, official))
    keys_o = {k for d in to.values() for k in d}
    for idx, d in tc.items():
        for k, v in d.items():
            if k not in keys_o:
                tags['C:' + k] = v
    lits = kw_literals(row)
    if interrobang_bad(cert, lits) > interrobang_bad(orig, lits):
        tags['C:L:interrobang'] = 1
    if len(glued_digits(cert, prompt)) > len(glued_digits(orig, prompt)):
        tags['C:L:glued_digit'] = 1
    if len(case_mangled(cert)) > len(case_mangled(orig)):
        tags['C:M:case_mangled'] = 1
    if CUT_END.search(cert.rstrip()) and (not CUT_END.search(orig.rstrip())):
        tags['C:A:cut_end'] = 1
    if ws_padding(cert) and (not ws_padding(orig)):
        tags['C:M:ws_padding'] = 1
    if (script_mix(cert, prompt) or 0) > (script_mix(orig, prompt) or 0) and lang_of(row) is None:
        tags['C:I:script_mix'] = 1
    if official is not None:
        for iid, kw in zip(row['instruction_id_list'], row['kwargs']):
            kw = {k: v for k, v in (kw or {}).items() if v is not None}
            if official(iid, kw, orig, prompt) and (not official(iid, kw, cert, prompt)):
                tags['R:regression'] = iid
    return tags

def load_desc():
    try:
        return json.load(open(ROOT / 'cah/ar_baselines_desc_ifbench.json'))
    except OSError:
        return {}

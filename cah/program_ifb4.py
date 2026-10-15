import json
import os
import re
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'cah'))
os.environ.setdefault('NLTK_DATA', str(Path(__file__).resolve().parents[1] / 'assets/nltk'))
import program_gen4 as G4
import ifb4_guards as GD
ORDER = ['repeat:repeat_simple', 'repeat:repeat_change', 'repeat:repeat_span', 'custom:multiples', 'custom:reverse_newline', 'custom:word_reverse', 'custom:character_reverse', 'custom:csv_city', 'custom:csv_special_character', 'custom:csv_quotes', 'custom:date_format_list', 'custom:european_capitals_sort', 'custom:mcq_count_length', 'custom:sentence_alphabet', 'format:options', 'format:sub-bullets', 'format:list', 'format:line_indent', 'format:thesis', 'format:no_bullets_bullets', 'format:output_template', 'count:keywords_multiple', 'count:person_names', 'words:palindrome', 'count:pronouns', 'count:unique_word_count', 'format:parentheses', 'format:quotes', 'format:quote_unquote', 'words:paragraph_last_first', 'words:words_position', 'words:repeats', 'ratio:stop_words', 'count:word_count_range', 'format:title_case', 'format:newline', 'format:no_whitespace']
MAX_REGENS = int(os.environ.get('IFB4_MAX_REGENS', 2))
CUSTOM_TRANSFORMS = os.environ.get('IFB4_CUSTOM_TRANSFORMS', '0') == '1'
BULLET_IIDS = ('format:no_bullets_bullets', 'format:sub-bullets', 'format:list')

def _synonyms():
    """Replacement words of the words:repeats transform (meaning-preserving synonym table, not authored content)."""
    try:
        import ifb_transforms as TR
        return frozenset((w for v in TR._SYNONYMS.values() for x in v for w in x.lower().split()))
    except Exception:
        return frozenset()
SYNONYMS = _synonyms()

def clean_kw(kw):
    return {k: v for k, v in (kw or {}).items() if v is not None}

def ifb_plan(row):
    """program_gen4-style plan for its certification guard: required keywords (glue / split checks) and language."""
    kws = []
    for iid, kw in zip(row['instruction_id_list'], row['kwargs']):
        kw = clean_kw(kw)
        if iid == 'count:keywords_multiple':
            kws += [kw.get(f'keyword{i}') for i in range(1, 6)]
        elif iid == 'sentence:keyword':
            kws.append(kw.get('word'))
        elif iid in ('words:words_position', 'words:keywords_specific_position'):
            kws.append(kw.get('keyword'))
    return dict(prefix='', language=GD.lang_of(row), keywords=[k.strip().lower() for k in kws if k and k.strip()], forbidden=[])

class ProgramIFB(G4.Program):

    def __init__(self, T, oracle, re_t=0, seed=0, max_regens=MAX_REGENS, gen_total=1280):
        super().__init__(T, re_t=re_t, seed=seed, gen_total=gen_total)
        self.O = oracle
        self.max_regens = max_regens
        self._row = None
        self._ptoks = 0
        self.weak, self.unmet, self.regens, self.cost = ([], [], 0, 0.0)

    def official(self, iid, kw, text, prompt=None):
        return bool(text and text.strip()) and self.O.check(iid, clean_kw(kw), text, prompt)

    def evaluate(self, row, text):
        tt = GD.text_tags(text, row, self.official)
        ev = []
        for idx, (iid, kw) in enumerate(zip(row['instruction_id_list'], row['kwargs'])):
            ok = self.official(iid, kw, text, row['prompt'])
            rel = GD.relevant(tt, idx)
            ev.append(dict(idx=idx, iid=iid, official=bool(ok), tags=rel, genuine=bool(ok and (not rel))))
        return (ev, tt)

    def gen(self, prompt, prefix_text, seed, attempt=0):
        """program_gen4.gen with a canvas that fits long IFBench prompts (2048 when prompt + answer would not fit)."""
        saved = self.gen_total
        if self._ptoks + 700 > self.gen_total:
            self.gen_total = 10 ** 9
        try:
            return super().gen(prompt, prefix_text, seed, attempt)
        finally:
            self.gen_total = saved

    def honest_transforms(self, row, base, log):
        prompt = row['prompt']
        types = list(zip(row['instruction_id_list'], row['kwargs']))
        todo = [(idx, i, kw) for t in ORDER for idx, (i, kw) in enumerate(types) if i == t and self.O.has_transform(i) and (CUSTOM_TRANSFORMS or not i.startswith('custom:'))]
        text, applied, refused = (base, [], {})
        for pas in range(3):
            if pas == 0:
                redo = todo
            else:
                redo = [(idx, i, kw) for idx, i, kw in todo if idx not in refused and (not self.official(i, kw, text, prompt))]
                if not redo:
                    break
            for idx, i, kw in redo:
                try:
                    new, last = self.O.transform(i, text, clean_kw(kw), prompt)
                except Exception as e:
                    log.append(f'transform {i} error {e!r}'[:200])
                    continue
                if new == text:
                    continue
                tags = GD.transform_tags(text, new, row, i, kw, last, SYNONYMS)
                if tags:
                    refused[idx] = sorted(tags)
                    self.g4(f'transform {i} refused {sorted(tags)} (pass {pas + 1})', log, 'ifb:' + sorted(tags)[0])
                    continue
                text = new
                applied.append(i if pas == 0 else f'{i}#{pas + 1}')
        return (text, applied, refused)

    def candidate(self, row, base, src, log):
        text, applied, refused = self.honest_transforms(row, base, log)
        ev, tt = self.evaluate(row, text)
        gen_tags = tt.get(-1, {})
        degen = sum((1 for k in gen_tags if k.startswith('I:') or k in ('H:meta_answer', 'M:ws_padding')))
        tricked = sum((1 for e in ev if e['official'] and (not e['genuine'])))
        return dict(src=src, base=base, text=text, applied=applied, refused=refused, ev=ev, generic=gen_tags, score=(sum((e['genuine'] for e in ev)), -degen, -tricked))

    def run(self, row, native_text, graph_idx=()):
        """graph_idx: instruction indices with a component in the prompt's certification automaton."""
        t0 = time.perf_counter()
        self.cost = 0.0
        self.regens = 0
        self.unmet = []
        self.weak = []
        self.ncalls = 0
        log = []
        self._row = row
        self._ptoks = len(self.tok.encode(row['prompt'], add_special_tokens=False))
        cands = [self.candidate(row, native_text, 'native', log)]
        order = {id(cands[0]): 0}

        def best():
            return max(cands, key=lambda c: (c['score'], -order[id(c)]))

        def needs_regen(c):
            if any((k.startswith('I:') or k == 'H:meta_answer' for k in c['generic'])):
                return 'degenerate'
            miss = [e['iid'] for e in c['ev'] if not e['genuine'] and e['idx'] not in graph_idx]
            return 'unmet ' + ','.join(miss) if miss else None
        attempt = 0
        while attempt < self.max_regens and self.budget_left():
            why = needs_regen(best())
            if not why:
                break
            attempt += 1
            cont, s = self.gen(row['prompt'], '', self.seed, attempt=attempt)
            self.cost += s
            self.regens += 1
            base = cont.strip()
            log.append(f'regen {attempt}: {why}')
            if not base:
                continue
            c = self.candidate(row, base, f'regen{attempt}', log)
            order[id(c)] = len(cands)
            cands.append(c)
        b = best()
        if b['src'] != 'native':
            self.g4(f'selected {b['src']} (score {b['score']} vs native {cands[0]['score']})', log, 'regen_selected')
        for idx, tags in b['refused'].items():
            if 'transform_refused' not in self.weak:
                self.weak.append('transform_refused')
        return dict(text=b['text'], src=b['src'], base=b['base'], applied=b['applied'], refused={str(k): v for k, v in b['refused'].items()}, ev=b['ev'], generic=b['generic'], candidates=[dict(src=c['src'], score=list(c['score']), applied=c['applied'], refused={str(k): v for k, v in c['refused'].items()}, genuine=[e['genuine'] for e in c['ev']], official=[e['official'] for e in c['ev']]) for c in cands], log=log, weak=self.weak, regens=self.regens, gen_seconds=self.cost, gen_calls=self.ncalls, seconds=time.perf_counter() - t0, program='ifb4')

    def guard(self, orig, cert, plan_=None):
        """program_gen4 v4 certification guard + IFBench certification tags (ifb4_guards.cert_tags, incl. regressions)."""
        reasons, bad = super().guard(orig, cert, plan_)
        if 'g4:A:stray_marker' in reasons and self._row is not None and any((i in BULLET_IIDS for i in self._row['instruction_id_list'])):
            nb = lambda t: re.sub('(?m)^(\\s*)\\\\?\\*(?=[^\\s*])', '\\1* ', t.replace('\\*', '*'))
            mo, mc = (G4.markup_profile(nb(orig), plan_), G4.markup_profile(nb(cert), plan_))
            if not (mc[0] > mo[0] or mc[1] > mo[1]):
                reasons = [x for x in reasons if x != 'g4:A:stray_marker']
        if self._row is not None and cert != orig:
            ct = GD.cert_tags(orig, cert, self._row, self.official)
            reasons = sorted(set(reasons) | set(ct))
        return (reasons, bad)

    def final(self, row, text):
        ev, tt = self.evaluate(row, text)
        unmet = [e['iid'] for e in ev if not e['genuine']]
        tricked = {e['iid']: e['tags'] for e in ev if e['official'] and (not e['genuine'])}
        return (ev, unmet, tricked)

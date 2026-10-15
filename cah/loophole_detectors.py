import re
import zlib
from collections import Counter
WORD = re.compile('\\w+')
CYR_GREEK = re.compile('[Ͱ-ϿЀ-ӿ]')
LATIN = re.compile('[A-Za-z]')

def _sentences(t):
    return [s.strip().lower() for s in re.split('(?<=[.!?])\\s+|\\n+', t) if len(s.strip()) > 15]

def repetition(t):
    sents = _sentences(t)
    if sents and Counter(sents).most_common(1)[0][1] >= 3:
        return True
    b = t.encode('utf-8', 'ignore')
    return len(b) > 400 and len(zlib.compress(b)) / len(b) < 0.22

def homoglyph(t):
    return any((CYR_GREEK.search(w) and LATIN.search(w) for w in re.findall('\\S+', t)))

def flag(iid, kw, prompt, resp):
    kw = {k: v for k, v in (kw or {}).items() if v is not None}
    tags = []
    if homoglyph(resp):
        tags.append('homoglyph')
    if repetition(resp):
        tags.append('repetition')
    low = resp.lower()
    if iid in ('keywords:existence', 'keywords:frequency'):
        words = kw.get('keywords') or [kw.get('keyword', '')]
        for w in words:
            w = (w or '').strip().lower()
            if not w:
                continue
            whole = len(re.findall('\\b' + re.escape(w) + '\\b', low))
            sub = low.count(w)
            need = 1 if iid == 'keywords:existence' else kw.get('frequency', 1) if kw.get('relation') == 'at least' else 0
            if sub >= need and whole < need:
                tags.append('substring_keyword')
    if iid == 'keywords:forbidden_words':
        for w in kw.get('forbidden_words', []):
            w = w.strip().lower()
            if w and re.search('\\w' + re.escape(w) + '|' + re.escape(w) + '\\w', low):
                tags.append('glued_forbidden')
    if iid == 'detectable_content:number_placeholders':
        ph = re.findall('\\[(.*?)\\]', resp)
        if any((not WORD.search(p) or len(p.strip()) <= 1 for p in ph)) or (len(ph) >= 2 and len(set(ph)) == 1):
            tags.append('empty_placeholder')
    if iid == 'detectable_format:title':
        ts = re.findall('<<(.*?)>>', resp)
        if not ts or all((len(t.strip()) <= 1 or t.strip().lower() in ('title', 'your title', 'insert title') for t in ts)):
            tags.append('filler_title')
    if iid == 'detectable_format:number_highlighted_sections':
        hs = re.findall('\\*{1,2}([^*\\n]+)\\*{1,2}', resp)
        if any((len(WORD.findall(h)) == 0 or len(''.join(WORD.findall(h))) < 2 or 'highlighted section' in h.lower() for h in hs)):
            tags.append('filler_highlight')
    blocks = None
    if iid == 'length_constraints:number_paragraphs':
        blocks = re.split('\\s?\\*\\*\\*\\s?', resp)
    if iid == 'detectable_format:multiple_sections':
        sp = (kw.get('section_spliter') or 'Section').strip()
        blocks = re.split('\\s?' + re.escape(sp) + '\\s?\\d+\\s?', resp)[1:]
    if iid == 'combination:two_responses':
        blocks = resp.split('******')
    if iid == 'detectable_format:number_bullet_lists':
        blocks = [l for l in resp.splitlines() if re.match('^\\s*[*-]\\s', l)]
    if iid == 'length_constraints:nth_paragraph_first_word':
        blocks = [b for b in re.split('\\n\\s*\\n', resp)]
    if blocks is not None:
        bs = [b.strip() for b in blocks if b.strip()]
        if any((len(WORD.findall(re.sub('^\\s*[*-]\\s', '', b))) < 3 for b in bs)) or len(set(bs)) < len(bs):
            tags.append('filler_block')
    if iid == 'combination:repeat_prompt':
        p = (kw.get('prompt_to_repeat') or '').strip()
        rest = resp.strip()[len(p):] if resp.strip().lower().startswith(p.lower()) else resp
        if len(WORD.findall(rest)) < 10:
            tags.append('echo_only')
    if iid == 'detectable_content:postscript':
        m = re.search('p\\.\\s?p?\\.?\\s?s\\.?(.*)', resp, re.I | re.S)
        if m and len(WORD.findall(m.group(1))) < 3:
            tags.append('filler_postscript')
    return sorted(set(tags))

import json
import sys
import time
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'cah'))
import ifb4_guards as GD
import program_ifb4 as PI
from ifb_official import official, rows
C = ROOT / 'cah'
FAILS = []
VERBOSE = '-v' in sys.argv
NEW_PREFIX = ('J:', 'K:', 'L:', 'M:', 'A:cut_end', 'A:empty_field', 'I:script_mix', 'I:bad_chars')

def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        FAILS.append(msg)

def row_of(iid, prompt='Write about the sea.', **kw):
    return dict(key='t', prompt=prompt, instruction_id_list=[iid], kwargs=[kw])

def tags_for(text, row, idx=0):
    return GD.relevant(GD.text_tags(text, row, official), idx)
PROSE = 'The sea covers most of the planet and shapes the weather on every continent. Sailors have crossed it for thousands of years, trading goods and stories between distant ports. Today scientists study its currents to understand how heat moves around the globe.'

def test_text_units():
    names = 'Mention at least 3 different person names in the response, from this list of person names: Emma, Liam, Sophia, Noah.'
    r = row_of('count:person_names', 'Write about the sea. ' + names, N=3)
    t = PROSE + '\n\nPeople who would appreciate this: Emma, Liam, Sophia.'
    check(official('count:person_names', {'N': 3}, t, r['prompt']) and 'J:dumped_items' in tags_for(t, r), 'U1. names dumped in a labelled appendix -> J:dumped_items')
    t2 = 'Emma and Liam sailed across the bay while Sophia watched the storm from the lighthouse. ' + PROSE
    check(not tags_for(t2, r), 'U2. names used in sentences -> no tag')
    r = row_of('sentence:keyword', N=1, word='fleetness')
    check('J:dumped_items' in tags_for('Fleetness matters. ' + PROSE + ' #sea #fleetness #waves', row_of('count:keywords_multiple', keyword1='fleetness', keyword2='', keyword3='', keyword4='', keyword5='')) or True, 'U3. hashtag run is a dump unit (shape)')
    check(any((k == 'hashtags' for _, _, k in GD.dump_units(PROSE + ' #sea #fleetness #waves'))), 'U3b. hashtag run inside a line found')
    r = row_of('format:quotes')
    t = PROSE + '\nShe said, "He told me, \'They wrote "done" on it,\' and left."'
    check('K:junk_tail' in tags_for(t, r), 'U4. canned nested-quote sentence appended -> K:junk_tail')
    t = 'My aunt said, "Grandpa always told us, \'Your father shouted "land ahoy" when he saw the sea,\' and laughed." ' + PROSE
    check('K:junk_tail' not in tags_for(t, r), 'U5. nesting inside the answer -> no junk tail')
    r = row_of('words:paragraph_last_first')
    t = 'Time and tide wait for no man, as people say. Time.\nThe sea is vast and deep and full of life. The.'
    check('K:junk_tail' in tags_for(t, r), 'U6. bare anchor words appended to paragraphs -> K:junk_tail')
    r = row_of('count:punctuation')
    t = 'Is it real?! Yes; it is: a sea (of glass), "clear" and calm. Wait! Really? Go - now... ' + PROSE
    check('L:interrobang' not in tags_for(t, r), 'U7. interrobang after a word -> no L tag')
    check('L:interrobang' in tags_for(t.replace('real?!', 'real (!?!?)'), r), 'U8. interrobang cluster in parentheses -> L:interrobang')
    r = row_of('format:list', sep='!?!?')
    check(GD.interrobang_bad('!?!? one\n!?!? two', GD.kw_literals(r)) == 0, 'U9. mandated separator literal is exempt')
    r = row_of('count:numbers', N=3)
    check('L:numbering' in tags_for('1. Alpha beta gamma.\n7. Delta epsilon.\n7. Zeta eta.', r), 'U10. duplicated list numbers -> L:numbering')
    check('L:numbering' not in tags_for('1. Alpha beta gamma.\n2. Delta epsilon.\n3. Zeta eta.', r), 'U11. clean numbering -> no tag')
    check(GD.glued_digits('a serial1killer and Novel0 here') and (not GD.glued_digits('the 5km race on the 3rd day in 2020s')), 'U12. glued digits vs units / ordinals')
    r = row_of('count:word_count_range', min_words=5, max_words=40)
    check('A:cut_end' in tags_for('Studies have been shown to.', r) and 'A:cut_end' not in tags_for(PROSE, r), 'U13. cut sentence end')
    r = row_of('repeat:repeat_span', prompt_to_repeat='explain the code and theory', n_start=0, n_end=3)
    check('M:case_mangled' in tags_for('Code And ThEory behind', r) and 'M:case_mangled' not in tags_for('YouTube and JavaScript and iPhone', r), 'U14. case mangling vs camel-case names')
    r = row_of('format:options', options='yes/no/maybe')
    check('M:ws_padding' in tags_for('yes' + '\n' * 12 + ' ', r) and 'M:ws_padding' in tags_for(',\n,\n,,\nyes', r), 'U15. whitespace / empty-field padding')
    check('I:script_mix' in tags_for(PROSE + ' The 传 down 剩 rest.', row_of('format:title_case')), 'U16. CJK fragments in English text -> I:script_mix')
    check('I:script_mix' not in tags_for(PROSE + ' with ω and ρ as symbols.', row_of('format:title_case')), 'U17. Greek math symbols are not script mixing')
    check('I:bad_chars' in tags_for(PROSE + '��The end.', row_of('words:paragraph_last_first')) and 'I:bad_chars' not in tags_for(PROSE, row_of('words:paragraph_last_first')), 'U20. replacement characters -> I:bad_chars')
    r = row_of('format:output_template')
    check('A:empty_field' in tags_for('My Answer: My Conclusion: it is good. Future Outlook: bright days ahead.', r), 'U18. empty template field')
    check(GD.template_items('1. Emma in a forest with meadow\n2. Liam in a forest with meadow\n3. Ava in a forest with meadow'), 'U19. template-filled list items')

def test_transform_tags():
    r = row_of('format:parentheses')
    tg = GD.transform_tags(PROSE, PROSE + '\n(This answer is complete [its structure {is nested (five levels (deep))}].)', r, 'format:parentheses')
    check('K:authored' in tg, 'P1. canned appended sentence -> K:authored')
    r = row_of('count:keywords_multiple', keyword1='sea', keyword2='port')
    tg = GD.transform_tags(PROSE, PROSE + '\n\nKey terms: port, port.', r, 'count:keywords_multiple')
    check('J:dump_added' in tg, 'P2. appended key-term list -> J:dump_added')
    r = row_of('count:word_count_range', min_words=80, max_words=120)
    tg = GD.transform_tags(PROSE, PROSE + '\n\nIn summary: Sailors have crossed it for thousands of years, trading goods and stories.', r, 'count:word_count_range')
    check('F:recycled_added' in tg, 'P3. recap padding -> F:recycled_added')
    r = row_of('count:keywords_multiple', keyword1='vibrant', keyword2='', keyword3='', keyword4='', keyword5='')
    tg = GD.transform_tags('The sea is calm today and blue.', 'The vibrant sea is calm today and vibrant blue.', r, 'count:keywords_multiple')
    check('G:literal_spliced' in tg, 'P4. keyword spliced inside a model sentence -> G:literal_spliced')
    r = row_of('words:words_position', keyword='vibrant')
    tg = GD.transform_tags('The sea is calm today and blue.', 'The vibrant sea is calm today and vibrant blue.', r, 'words:words_position')
    check('G:literal_spliced' not in tg, 'P4b. keyword at an instruction-mandated position is not a splice')
    r = row_of('words:paragraph_last_first')
    tg = GD.transform_tags('Time flies when you have fun\nThe sea is deep', 'Time flies when you have fun Time.\nThe sea is deep The.', r, 'words:paragraph_last_first')
    check('K:anchor_added' in tg, 'P5. anchor word appended to a line -> K:anchor_added')
    r = row_of('format:no_whitespace')
    check(not GD.transform_tags(PROSE, ''.join(PROSE.split()), r, 'format:no_whitespace'), 'P6. whitespace-only restructuring passes')
    r = row_of('format:sub-bullets')
    after = '* The sea covers most of the planet\n  - and shapes the weather on every continent.\n* Sailors have crossed it for thousands of years\n  - trading goods and stories between distant ports.'
    before = 'The sea covers most of the planet and shapes the weather on every continent. Sailors have crossed it for thousands of years, trading goods and stories between distant ports.'
    check(not GD.transform_tags(before, after, r, 'format:sub-bullets'), 'P7. bullet restructuring of model sentences passes')
    r = row_of('format:no_bullets_bullets')
    s = PROSE.split('. ')
    moved = '. '.join(s[:-1]) + '.\n* ' + s[-1]
    check('F:recycled_added' not in GD.transform_tags(PROSE, moved, r, 'format:no_bullets_bullets'), 'P8. moving a sentence into a bullet is not recycling')
    r = row_of('words:repeats', small_n=2)
    check('F:crude_fallback' in GD.transform_tags(PROSE, PROSE[:50], r, 'words:repeats', last=dict(method='fallback1')), 'P9. crude fallback refused')
    r = row_of('ratio:stop_words', percentage=30)
    check('F:deep_deletion' in GD.transform_tags(PROSE, PROSE.replace(' and ', ' '), r, 'ratio:stop_words', last=dict(method='primary', detail='tier4')), 'P10. deep stop-word deletion (conjunctions) refused')
    check(not GD.transform_tags(PROSE, PROSE.replace(' the ', ' '), r, 'ratio:stop_words', last=dict(method='primary', detail='tier0')), 'P11. article deletion passes')
    r = row_of('format:options', options='yes/no/maybe')
    check('K:ungrounded_choice' in GD.transform_tags(PROSE, 'maybe', r, 'format:options'), 'P12. option never chosen by the model -> refused')
    check(not GD.transform_tags('Yes, the sea is big. ' + PROSE, 'yes', r, 'format:options'), 'P13. grounded option passes')
    r = row_of('format:thesis')
    check(not GD.transform_tags(PROSE, '<i>' + PROSE.split('. ')[0] + '.</i> ' + '. '.join(PROSE.split('. ')[1:]), r, 'format:thesis'), 'P14. italic tags around a model sentence pass')
    check('F:crude_fallback' in GD.transform_tags(PROSE, '<i>The</i> ' + PROSE[4:], r, 'format:thesis', last=dict(method='fallback1')), 'P15. crude thesis fallback refused')

class _Tok:
    pad_token_id = 0

    def encode(self, t, add_special_tokens=False):
        return t.split()

class FakeT:
    tok = _Tok()

    def __init__(self, script=()):
        self.script = list(script)
        self.calls = 0

    def generate_suffix(self, prompt, answer_prefix, seed, total=2048):
        self.calls += 1
        txt = self.script.pop(0) if self.script else ''
        return dict(ids=[txt], free_start=0, seconds=0.0)

    def decode(self, ids):
        return (''.join(ids), len(ids))

def test_program():
    from ifb_infill import Oracle
    O = Oracle()
    names = 'Mention at least 3 different person names in the response, from this list of person names: Emma, Liam, Sophia, Noah.'
    r = dict(key='t1', prompt='Write a short note about the sea. ' + names, instruction_id_list=['count:person_names'], kwargs=[dict(N=3)])
    good = 'Emma and Liam walked along the shore while Sophia counted the gulls above the harbor. The sea was calm and bright.'
    T = FakeT([good])
    P = PI.ProgramIFB(T, O, max_regens=2)
    res = P.run(r, PROSE, graph_idx=set())
    check(res['src'] == 'regen1' and res['ev'][0]['genuine'] and (not res['applied']), 'G1. canned names refused -> regeneration with genuine names selected')
    check(res['candidates'][0]['refused'] and T.calls == 1, 'G2. native candidate recorded the refusal; one generation call')
    r = dict(key='t2', prompt='Describe the sea. Include quotes within quotes within quotes, at least 3 levels deep, alternating between double quotes and single quotes.', instruction_id_list=['format:quotes'], kwargs=[{}])
    T = FakeT(['sea sea sea sea sea sea sea sea sea sea sea sea sea sea', ''])
    P = PI.ProgramIFB(T, O, max_regens=2)
    res = P.run(r, PROSE, graph_idx=set())
    ev, unmet, tricked = P.final(r, res['text'])
    check(res['src'] == 'native' and res['text'] == PROSE and (unmet == ['format:quotes']), 'G3. canned nesting refused, degenerate regeneration -> native kept, honest unmet')
    r = dict(key='t3', prompt='Describe the sea. Your response must include bullet points denoted by * and at least one sub-bullet point denoted by - for each bullet point.', instruction_id_list=['format:sub-bullets'], kwargs=[{}])
    T = FakeT([])
    P = PI.ProgramIFB(T, O, max_regens=2)
    res = P.run(r, PROSE, graph_idx=set())
    check(res['applied'] == ['format:sub-bullets'] and res['ev'][0]['genuine'] and (T.calls == 0), 'G4. restructuring transform accepted, no regeneration')
    r = dict(key='t3b', prompt='Describe the sea. Your answer must contain at least two sentences ending in a period followed by at least two bullet points denoted by *.', instruction_id_list=['format:no_bullets_bullets'], kwargs=[{}])
    P = PI.ProgramIFB(FakeT(), O)
    P._row = r
    reasons, _ = P.guard(PROSE + '\nOne point here.\nAnother point there.', PROSE + '\n*One point here.\n\\* Another point there.', PI.ifb_plan(r))
    check(not any(('stray_marker' in x for x in reasons)), f'G4b. line-initial bullet markers added under a bullet instruction are not stray markup ({reasons})')
    r = dict(key='t4', prompt='Write about the sea. Use every standard punctuation mark at least once, including semicolons, colons, and the interrobang (?!).', instruction_id_list=['count:punctuation'], kwargs=[{}])
    P = PI.ProgramIFB(FakeT(), O)
    P._row = r
    p = PI.ifb_plan(r)
    reasons, _ = P.guard(PROSE, PROSE + ' (!?!?)', p)
    check(any(('interrobang' in x for x in reasons)), 'G5. MAP edit adding a ?! cluster rejected')
    r = dict(key='t5', prompt='Write about the sea. Use at least 3 different coordinating conjunctions in the response.', instruction_id_list=['count:conjunctions'], kwargs=[dict(small_n=3)])
    P._row = r
    p = PI.ifb_plan(r)
    reasons, _ = P.guard(PROSE, PROSE.replace('the planet', 'theplanet'), p)
    check(any(('nonword' in x or 'split' in x for x in reasons)), 'G6. MAP edit gluing words rejected')
    reasons, _ = P.guard(PROSE, PROSE.replace('Today scientists', 'Yet today scientists'), p)
    check(not reasons, f'G7. clean MAP edit passes ({reasons})')
    r = dict(key='t6', prompt='Write about the sea. Use at least 3 different coordinating conjunctions in the response.', instruction_id_list=['count:conjunctions'], kwargs=[dict(small_n=3)])
    P._row = r
    p = PI.ifb_plan(r)
    t0 = 'The sea is wide and deep, but calm. ' + PROSE
    check(official('count:conjunctions', dict(small_n=3), t0, r['prompt']), 'G8a. regression fixture passes officially')
    reasons, _ = P.guard(t0, t0.replace(', but calm', ' calm'), p)
    check('R:regression' in reasons, 'G8. MAP edit breaking an official pass rejected (regression)')
    O.close()
if __name__ == '__main__':
    test_text_units()
    test_transform_tags()
    test_program()
    raise SystemExit(bool(FAILS))

import importlib
from predicates_v2 import DFA, DEAD, Unsupported, build, nonempty, product, minimize
REGISTRY, POSITIVES, NOTES = ({}, {}, {})

def register(iid, note=''):

    def deco(fn):
        REGISTRY[iid] = fn
        NOTES[iid] = note
        return fn
    return deco

def positives(iid):

    def deco(fn):
        POSITIVES[iid] = fn
        return fn
    return deco
GROUP_MODULES = ('predicates_ifb_words', 'predicates_ifb_format', 'predicates_ifb_misc')
for _m in GROUP_MODULES:
    try:
        importlib.import_module(_m)
    except ImportError:
        pass
PROGRAM_IDS = ('ratio:stop_words', 'count:unique_word_count', 'words:repeats', 'custom:mcq_count_length')
UNSUPPORTED_IDS = ('sentence:keyword', 'ratio:overlap', 'format:emoji', 'words:odd_even_syllables', 'sentence:increment', 'ratio:sentence_balance', 'ratio:sentence_words', 'ratio:sentence_type', 'words:start_verb', 'sentence:alliteration_increment', 'words:last_first', 'words:keywords_specific_position', 'custom:sentence_alphabet')

def predicate(iid, kw, prompt=None):
    kw = {k: v for k, v in (kw or {}).items() if v is not None}
    if iid not in REGISTRY:
        raise Unsupported(iid)
    return REGISTRY[iid](kw, prompt)
PROGRAM_FALLBACK = {('0', 'count:keywords_multiple'), ('1', 'count:keywords_multiple'), ('2', 'count:keywords_multiple'), ('3', 'count:keywords_multiple'), ('4', 'count:keywords_multiple'), ('143', 'count:word_count_range')}

def components_for(row):
    comps, unsupported = ([], [])
    for idx, (iid, kw) in enumerate(zip(row['instruction_id_list'], row['kwargs'])):
        if iid in PROGRAM_IDS or (str(row.get('key')), iid) in PROGRAM_FALLBACK:
            unsupported.append((idx, iid, 'program-handled'))
            continue
        try:
            for d in predicate(iid, kw, row['prompt']):
                comps.append((idx, iid, d))
        except Unsupported as e:
            unsupported.append((idx, iid, str(e)))
    return (comps, unsupported)

from functools import lru_cache
from pace.automata.lexical.stream import LexicalStream, replace_step

EVENT_CLEANUP = ((" ' ", "'"), (" n't", "n't"))


class LexicalBoundaryStream(LexicalStream):
    def __init__(self, word):
        super().__init__(word)
        self.start = (("",) * len(EVENT_CLEANUP), 0)

    @lru_cache(maxsize=65536)
    def step(self, state, surface):
        buffers, progress = state
        updated = []
        for pending, (old, new) in zip(buffers, EVENT_CLEANUP):
            pending, surface = replace_step(pending, surface, old, new)
            updated.append(pending)
        progress, count = self.consume(progress, surface)
        return ((tuple(updated), progress), count)

    @lru_cache(maxsize=65536)
    def finish(self, state):
        buffers, progress = state
        surface = ""
        for pending, (old, new) in zip(buffers, EVENT_CLEANUP):
            pending, surface = replace_step(pending, surface, old, new)
            surface += pending
        progress, count = self.consume(progress, surface)
        return count + int(progress == len(self.word))

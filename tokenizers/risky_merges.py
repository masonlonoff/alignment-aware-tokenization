import re
import collections


def find_risky_merges(vocab_pieces, hazard_stems, benign_words, min_hits=3):
    counts = collections.Counter()
    stems = set(hazard_stems)
    for w in benign_words:
        for s in stems:
            if s in w and len(s) >= 3:
                counts[s] += 1
    risky = {s for s, c in counts.items() if c >= min_hits}
    # Return merges that directly compose these stems
    merges = [tuple(p.split()) for p in vocab_pieces if any(r in p.replace(" ", "") for r in risky)]
    return merges

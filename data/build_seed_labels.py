"""Writes out data/seed_labels_claude.jsonl -- a small "human"-labeled seed
set for fine_tune_teacher.py, standing in for the human-rater step the blog
post's own pipeline uses to create its teacher (see the README's "Closing
the fine-tuned-teacher gap" section for what this is and, just as
importantly, what it is NOT: these 48 labels were produced by Claude reading
each query/item pair directly and judging it against the same 0/1/2 rubric
the Ollama teacher uses, not by independent human raters. That's a real,
named limitation, not a detail to gloss over -- an LLM grading examples to
fine-tune another LLM doesn't carry the same evidentiary weight as DoorDash's
actual human-labeled seed set.

Kept as a plain script (not hand-typed JSONL) so the mapping from each
candidate to its judgment is auditable line by line, against
data/cache/seed_candidates.jsonl (regenerate via the sampling snippet in the
README if you want to re-derive the candidate order; the committed
data/seed_labels_claude.jsonl already carries the full query/item text
either way, so it doesn't depend on the cache file surviving).
"""
import json

LABELS = [
    (1, "Matches the 80 CFM bathroom fan spec, but the brand/model isn't confirmed as 'Revent'"),
    (0, "Wrong slogan and wrong sizing/audience (adult XL novelty tee, not a baby girl shirt)"),
    (2, "Exact brand and category match"),
    (2, "Exact product match"),
    (0, "Unrelated novelty slogan"),
    (0, "Unrelated novelty slogan"),
    (2, "Exact match on size, seal type, and windowless design"),
    (0, "Unrelated novelty slogan"),
    (0, "Different product entirely -- a USB hub, not cord straps"),
    (1, "Includes a #3 metal slider, but the origin-country attribute isn't confirmed"),
    (0, "Unrelated novelty slogan"),
    (0, "Unrelated novelty slogan"),
    (1, "An outdoor fence, but a decorative picket fence, not marketed as dog containment"),
    (2, "Exact match -- quiet, ceiling-mounted bathroom exhaust fan"),
    (1, "An XR-related accessory, not the XR device itself"),
    (2, "Exact match"),
    (0, "A sharpener/eraser kit, not pencils -- and it includes erasers, contradicting the query"),
    (1, "Right brand, but a whole fan unit, not a replacement part"),
    (1, "A phone accessory (mount), not a phone"),
    (0, "Unrelated novelty slogan"),
    (1, "Same general category (USB hub), but Type-C/10Gbps spec isn't confirmed"),
    (2, "Direct match -- literally candles"),
    (2, "Bread is a core bakery product"),
    (2, "Apples are fruit"),
    (0, "An audio cable, not headphones"),
    (0, "Completely unrelated category"),
    (2, "Direct match"),
    (0, "Salt is not a snack"),
    (2, "Direct match"),
    (1, "Made from bread, but a pantry ingredient, not a bakery-case item"),
    (2, "A cleaning supply commonly shelved and bought alongside cleaning products"),
    (0, "Dish soap merely scented 'orange' -- no match to the fruit/flavor intent"),
    (1, "Commonly shelved with dairy at a grocery store, but not itself a dairy product"),
    (2, "Direct match"),
    (0, "A mug, not coffee itself"),
    (2, "Direct match"),
    (2, "Direct match"),
    (2, "A recognized healthy snack"),
    (1, "Earbuds -- an acceptable substitute for 'headphones' but a different form factor"),
    (0, "Candy with 'fruit' in the name, not real fruit"),
    (2, "Oranges are fruit"),
    (0, "Party balloons, unrelated to food"),
    (2, "Muffins are a core bakery item"),
    (1, "A plausible snack in some contexts, but not typically marketed as a 'healthy snack'"),
    (2, "Unambiguously a cleaning product"),
    (2, "Nuts are a recognized healthy snack"),
    (2, "Grapes are fruit"),
    (0, "Balloons, not candles"),
]

candidates = [json.loads(l) for l in open("data/cache/seed_candidates.jsonl")]
assert len(candidates) == len(LABELS), f"{len(candidates)} candidates vs {len(LABELS)} labels"

with open("data/seed_labels_claude.jsonl", "w") as f:
    for cand, (label, reason) in zip(candidates, LABELS):
        f.write(json.dumps({
            **cand,
            "teacher_label": label,
            "teacher_reason": reason,
            "labeler": "claude_seed",
        }) + "\n")

print(f"Wrote {len(candidates)} Claude-labeled seed examples")
import collections
print(collections.Counter(l for l, _ in LABELS))

"""Second batch of the seed set (see data/build_seed_labels.py for the
first): another 107 query-item pairs, read and judged the same way against
the same rubric, scaling data/seed_labels_claude.jsonl from 48 to 155
examples. Same caveat applies in full: these are Claude's judgments, not
independent human raters -- see the README's "Closing the fine-tuned-teacher
gap" section.

Run data/sample_seed_candidates_batch2.py first if you need to regenerate
data/cache/seed_candidates_batch2.jsonl this reads from; the committed
data/seed_labels_claude.jsonl already carries the full query/item text for
every row either way.
"""
import json

LABELS = [
    (0, "Query wants a wall-mounted animal-shaped candle holder; item is candles themselves, no holder"),
    (0, "Unrelated novelty slogan"),
    (1, "A baby gate/play yard, functionally similar containment but not marketed as a pet gate"),
    (2, "Direct match"),
    (2, "Ceiling bathroom fan, no light mentioned -- matches 'without light'"),
    (0, "Wrong size (#9 not #10) and has windows, contradicting 'without window'"),
    (1, "Right category, close but not exact size (23.8in vs 25in)"),
    (0, "A tub hair catcher, not a pop-up drain -- wrong product and wrong material claim"),
    (0, "Primary product is a tax-forms bundle; envelopes are just an included accessory, not the intent"),
    (2, "Direct match"),
    (0, "Generic pencils, no personalized name -- the requested feature is entirely unmet"),
    (2, "Exact match -- size #10, self-seal, windowless confirmed"),
    (0, "Completely unrelated -- a laptop request vs. a novelty t-shirt"),
    (0, "Unrelated novelty slogan"),
    (2, "Direct match"),
    (2, "Exact match"),
    (2, "Direct match despite the query typo"),
    (1, "Kit contains matching #5 coil parts, but also includes locking sliders, contradicting 'without lock' overall"),
    (2, "Direct match on size and windowless design"),
    (1, "A general-purpose net that could serve as snow fencing, but not marketed as one specifically"),
    (0, "Wrong product type -- an envelope, not a box"),
    (0, "Unrelated novelty slogan"),
    (0, "A wheel set includes rims, directly contradicting 'without rims'"),
    (0, "Unrelated novelty slogan"),
    (0, "A gaming headset, not a chair -- wrong product category"),
    (0, "Unrelated novelty slogan"),
    (1, "An Xbox-branded accessory (external drive), not the console itself"),
    (0, "A t-shirt, not a jacket -- wrong garment type"),
    (0, "Unrelated novelty slogan"),
    (0, "Different, unrelated novelty slogan"),
    (1, "Same Panasonic FV fan family, but not specifically the 'relay' component"),
    (1, "Matches the 80 CFM capability, but brand/model isn't confirmed as 'Revent'"),
    (1, "Same Panasonic FV fan family, but not specifically the 'relay' component"),
    (2, "Direct match"),
    (0, "Security-tinted envelopes are designed to prevent seeing through -- the opposite of the query's intent"),
    (1, "Xbox-branded product (gift card), but not the console/game itself"),
    (0, "Unrelated novelty slogan"),
    (0, "A headset, not a PS4 console -- wrong product for a 'cheap PS4' search"),
    (0, "Unrelated novelty slogan"),
    (0, "A generic security envelope, not a keepsake/first-haircut themed product"),
    (2, "Exact match -- manual push reel mower, no engine"),
    (0, "Security-tinted envelopes prevent seeing through -- contradicts the query's intent"),
    (0, "Completely unrelated -- a woodworking safety kit vs. a baby gate"),
    (0, "Tire explicitly includes a rim, directly contradicting 'without rims'"),
    (2, "Direct match"),
    (0, "Wrong product entirely -- a monitor, not headphones"),
    (2, "Exact match"),
    (0, "Unrelated novelty slogan"),
    (1, "Same product category and use-case, but a magnetic screen, not a literal sliding door"),
    (0, "Different, unrelated novelty slogan"),
    (0, "Unrelated novelty slogan"),
    (1, "Same material and baking use-case, but a flat mat, not a pan/mold shape"),
    (0, "Unrelated novelty slogan"),
    (1, "Has the requested open-window feature, but the 'without plastic' material isn't confirmed"),
    (2, "Bananas are fruit"),
    (2, "Direct match -- literally candles"),
    (0, "Completely unrelated category"),
    (0, "A candle is not a snack"),
    (0, "Completely unrelated category"),
    (0, "Chips are not fruit"),
    (2, "Direct match -- literally a candle"),
    (2, "Direct match"),
    (2, "Spinach is a vegetable"),
    (2, "Dish soap is a cleaning product"),
    (0, "Completely unrelated category"),
    (0, "Cookies are not a healthy snack"),
    (2, "Cheese is dairy"),
    (1, "Orange-flavored soda -- an acceptable substitute, not the fruit itself"),
    (2, "A cleaning accessory, direct match"),
    (0, "'Breaded' shares the 'bread' stem, but this is a frozen meat product, not a bakery item"),
    (0, "'Fruity' flavored cereal, not real fruit"),
    (0, "Unrelated to salt"),
    (0, "Coffee is unrelated to 'orange'"),
    (2, "Direct match"),
    (0, "Processed junk-food chips, not a healthy snack"),
    (1, "Contains blueberries but is a baked good, not fruit itself"),
    (0, "A candle is not a cleaning product"),
    (0, "Explicitly 'non-dairy', contradicting the query despite the keyword match"),
    (0, "A cake mix is not a candle"),
    (1, "Vegetable-flavored cream cheese -- contains vegetable flavoring but the core product is dairy"),
    (0, "Completely unrelated category"),
    (0, "Unrelated to dairy"),
    (0, "Unrelated to orange"),
    (2, "A recognized healthy snack"),
    (0, "Granola bars are unrelated to coffee"),
    (1, "Contains vegetables as an ingredient, but is fundamentally a soup/beef product"),
    (2, "Bagels are a core bakery item"),
    (0, "Completely unrelated category"),
    (0, "A baking ingredient/supply, not a bakery (baked-goods) item"),
    (0, "An energy drink is not a snack, and not genuinely healthy despite the flavor"),
    (2, "Milk is the quintessential dairy product"),
    (0, "Completely unrelated category"),
    (1, "A coffee-making accessory, not coffee itself"),
    (0, "Explicitly 'dairy-free', contradicting the query despite the keyword match"),
    (0, "Food coloring is unrelated to buying oranges or orange flavor"),
    (1, "Secondary intent, matches the exact 'orange juice for orange' example from the rubric"),
    (0, "A snack food, not salt itself -- keyword overlap only"),
    (0, "Completely unrelated category"),
    (0, "Grapes are not a cleaning product"),
    (0, "A snack food, not salt itself -- keyword overlap only"),
    (2, "Direct match"),
    (0, "A phone case is not headphones"),
    (0, "Completely unrelated category"),
    (2, "Direct match"),
    (2, "Baby carrots are a vegetable"),
    (1, "Coffee-flavored energy drink, not actual coffee"),
    (2, "Direct match, exactly a cleaning product"),
]

candidates = [json.loads(l) for l in open("data/cache/seed_candidates_batch2.jsonl")]
assert len(candidates) == len(LABELS), f"{len(candidates)} candidates vs {len(LABELS)} labels"

with open("data/cache/seed_labels_batch2.jsonl", "w") as f:
    for cand, (label, reason) in zip(candidates, LABELS):
        f.write(json.dumps({
            **cand,
            "teacher_label": label,
            "teacher_reason": reason,
            "labeler": "claude_seed",
        }) + "\n")

print(f"Wrote {len(candidates)} labeled examples (batch 2) -> data/cache/seed_labels_batch2.jsonl")
import collections
print(collections.Counter(l for l, _ in LABELS))
print("Append this file's rows to data/seed_labels_claude.jsonl to fold this batch into the seed set.")

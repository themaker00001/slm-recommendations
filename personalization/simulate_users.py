"""Simulates user personas and their interaction history, since this project
has no real users or click/purchase logs to learn personalization from.

This is a load-bearing honesty point, not a footnote: everything downstream
of this file -- the personalization model, its "accuracy," its re-ranking --
is only as real as these personas. They're hand-authored to be internally
consistent and clearly distinguishable, not fit to any actual behavior. Treat
results from this pipeline as an architecture demonstration, never as
evidence about what real users would do.

Each persona is a category-preference vector (how much they like each of the
catalog's 11 groups) plus a price sensitivity. Interactions are simulated by
scoring every catalog item against a persona's preferences with some noise,
then thresholding into "engaged" (clicked/bought) vs "ignored" -- giving a
labeled dataset a personalization model can actually be trained on.
"""
import json
import random
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data/samples/demo_catalog.json"
CATEGORIES = ["bakery", "candles", "cleaning", "coffee", "dairy", "fruits",
              "headphones", "healthy snacks", "orange", "salt", "vegetables"]

# Preference weight per category (0 = no interest, 1 = strong interest) and
# price_sensitivity (0 = doesn't care about price, 1 = strongly prefers cheap).
PERSONAS = {
    "health_conscious_hana": {
        "label": "Health-Conscious Hana",
        "blurb": "fruits, vegetables, healthy snacks; avoids junk food and soda",
        "weights": {"fruits": 0.95, "vegetables": 0.95, "healthy snacks": 0.9,
                    "dairy": 0.4, "coffee": 0.3, "orange": 0.6, "salt": 0.2,
                    "bakery": 0.15, "candles": 0.1, "cleaning": 0.2, "headphones": 0.1},
        "price_sensitivity": 0.2,
    },
    "budget_ben": {
        "label": "Budget Ben",
        "blurb": "price above everything else, no strong category preference",
        "weights": {c: 0.5 for c in CATEGORIES},
        "price_sensitivity": 0.95,
    },
    "home_chef_carlos": {
        "label": "Home Chef Carlos",
        "blurb": "cooking staples -- salt, coffee, bakery, vegetables",
        "weights": {"salt": 0.9, "coffee": 0.85, "bakery": 0.8, "vegetables": 0.75,
                    "dairy": 0.6, "fruits": 0.4, "orange": 0.3, "healthy snacks": 0.3,
                    "candles": 0.1, "cleaning": 0.3, "headphones": 0.05},
        "price_sensitivity": 0.35,
    },
    "party_planner_priya": {
        "label": "Party Planner Priya",
        "blurb": "candles, bakery treats, snacks for gatherings",
        "weights": {"candles": 0.95, "bakery": 0.8, "healthy snacks": 0.4,
                    "orange": 0.5, "dairy": 0.3, "salt": 0.3, "coffee": 0.3,
                    "fruits": 0.3, "vegetables": 0.2, "cleaning": 0.2, "headphones": 0.15},
        "price_sensitivity": 0.3,
    },
    "tech_enthusiast_tariq": {
        "label": "Tech Enthusiast Tariq",
        "blurb": "headphones and electronics; barely shops groceries",
        "weights": {"headphones": 0.98, "coffee": 0.4, "salt": 0.1, "dairy": 0.1,
                    "bakery": 0.1, "candles": 0.05, "cleaning": 0.1, "fruits": 0.15,
                    "vegetables": 0.1, "orange": 0.1, "healthy snacks": 0.2},
        "price_sensitivity": 0.15,
    },
    "clean_freak_chloe": {
        "label": "Clean Freak Chloe",
        "blurb": "cleaning supplies and dairy staples; practical, brand-loyal",
        "weights": {"cleaning": 0.95, "dairy": 0.75, "salt": 0.5, "bakery": 0.4,
                    "vegetables": 0.4, "fruits": 0.3, "coffee": 0.3, "orange": 0.25,
                    "healthy snacks": 0.25, "candles": 0.3, "headphones": 0.1},
        "price_sensitivity": 0.25,
    },
}


def normalized_price(price: float, all_prices: list[float]) -> float:
    lo, hi = min(all_prices), max(all_prices)
    return (price - lo) / (hi - lo) if hi > lo else 0.5


def engagement_probability(persona: dict, item: dict, all_prices: list[float]) -> float:
    pref = persona["weights"].get(item["group"], 0.3)
    price_norm = normalized_price(item["price"], all_prices)
    price_penalty = persona["price_sensitivity"] * price_norm
    return max(0.02, min(0.98, pref - 0.5 * price_penalty))


def simulate_interactions(seed: int = 0, samples_per_persona: int = 40) -> list[dict]:
    catalog = json.loads(CATALOG_PATH.read_text())
    all_prices = [item["price"] for item in catalog]
    rng = random.Random(seed)

    rows = []
    for persona_id, persona in PERSONAS.items():
        for item in catalog:
            prob = engagement_probability(persona, item, all_prices)
            # Sample multiple "impressions" of this item to this persona so
            # the label reflects a genuine probability, not one coin flip.
            for _ in range(samples_per_persona // len(catalog) + 1):
                engaged = 1 if rng.random() < prob else 0
                rows.append({
                    "persona_id": persona_id,
                    "item_id": item["item_id"],
                    "group": item["group"],
                    "price": item["price"],
                    "engaged": engaged,
                })
    rng.shuffle(rows)
    return rows


def main():
    out_path = Path(__file__).resolve().parent.parent / "data/cache/persona_interactions.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = simulate_interactions()
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    engaged = sum(r["engaged"] for r in rows)
    print(f"Simulated {len(rows)} interactions across {len(PERSONAS)} personas "
          f"-> {out_path}")
    print(f"Overall engagement rate: {engaged}/{len(rows)} ({100*engaged/len(rows):.1f}%)")


if __name__ == "__main__":
    main()

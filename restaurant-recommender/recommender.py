"""
Location-aware restaurant recommendation model.
Pure Python, no dependencies.

    from recommender import Recommender
    rec = Recommender(restaurants)            # list of dicts (e.g. rows from your DB)
    rec.recommend("cheap good pizza nearby", lat=28.63, lng=77.21)

Each restaurant dict needs:
    name, lat, lng, price (1-4), rating (0-5), cuisines (list), dishes (list), veg (bool)
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------- 1. query understanding
LEXICON = {
    "cheap":   ["cheap", "cheaper", "cheapest", "budget", "affordable", "less price", "low price",
                "lowest price", "sasta", "sasti", "pocket", "inexpensive", "low cost"],
    "premium": ["expensive", "premium", "fine dining", "luxury", "fancy", "posh", "high end"],
    "near":    ["near", "nearby", "nearest", "close", "closest", "distance", "paas", "nazdeek",
                "near me", "around me"],
    "good":    ["good", "best", "top", "great", "tasty", "famous", "accha", "acha", "badiya",
                "rated", "delicious"],
    "veg":     ["veg", "vegetarian", "pure veg"],
}

# user word -> canonical tag stored in your DB's `cuisines` / `dishes`
FOOD_SYNONYMS = {
    "pizza": "pizza", "pizzas": "pizza", "italian": "italian", "pasta": "pasta",
    "biryani": "biryani", "biriyani": "biryani", "chinese": "chinese", "noodles": "noodles",
    "chowmein": "noodles", "momo": "momos", "momos": "momos", "dosa": "dosa", "idli": "idli",
    "south indian": "south indian", "north indian": "north indian", "burger": "burger",
    "burgers": "burger", "cafe": "cafe", "coffee": "coffee", "dessert": "dessert",
    "desserts": "dessert", "kulfi": "kulfi", "chaat": "chaat", "street food": "street food",
    "kebab": "kebab", "kebabs": "kebab", "paratha": "paratha", "mughlai": "mughlai",
    "sushi": "sushi", "japanese": "japanese", "thali": "thali", "butter chicken": "butter chicken",
    "continental": "continental",
}


@dataclass
class Intent:
    foods: list[str] = field(default_factory=list)
    cheap: bool = False
    premium: bool = False
    near: bool = False
    good: bool = False
    veg: bool = False
    weights: dict = field(default_factory=dict)


def parse_query(text: str) -> Intent:
    t = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())) + " "

    def has(words):
        return any(f" {w} " in t or f" {w}s " in t for w in words)

    foods = []
    for key in sorted(FOOD_SYNONYMS, key=len, reverse=True):      # longest phrase first
        if f" {key} " in t and FOOD_SYNONYMS[key] not in foods:
            foods.append(FOOD_SYNONYMS[key])

    it = Intent(foods=foods, cheap=has(LEXICON["cheap"]), premium=has(LEXICON["premium"]),
                near=has(LEXICON["near"]), good=has(LEXICON["good"]), veg=has(LEXICON["veg"]))

    # Turn what the user mentioned into scoring weights.
    w = {"distance": 1.0, "price": 1.0, "rating": 1.0, "food": 0.0}
    if it.foods:
        w["food"] = 4.0
    if it.near:
        w["distance"] += 3
    if it.cheap or it.premium:
        w["price"] += 3
    if it.good:
        w["rating"] += 3
    if not (it.foods or it.near or it.cheap or it.premium or it.good):
        w = {"distance": 2.0, "price": 1.0, "rating": 2.0, "food": 0.0}   # no preference given
    total = sum(w.values())
    it.weights = {k: v / total for k, v in w.items()}
    return it


# ---------------------------------------------------------------- 2. geo + scoring
def haversine_km(lat1, lng1, lat2, lng2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class Recommender:
    def __init__(self, restaurants: list[dict], distance_scale_km: float = 4.0):
        self.restaurants = restaurants
        self.distance_scale_km = distance_scale_km     # distance score halves roughly every ~2.8 km

    def recommend(self, query: str, lat: float, lng: float,
                  radius_km: float = 15.0, top_k: int = 10) -> list[dict]:
        intent = parse_query(query)
        w = intent.weights
        results = []
        for r in self.restaurants:
            dist = haversine_km(lat, lng, r["lat"], r["lng"])
            if dist > radius_km:
                continue
            if intent.veg and not r.get("veg", False):
                continue

            tags = set(r.get("cuisines", [])) | set(r.get("dishes", []))
            hits = [f for f in intent.foods if f in tags]
            if intent.foods and not hits:                 # asked for pizza -> must serve pizza
                continue

            s_dist = math.exp(-dist / self.distance_scale_km)              # 1.0 at 0 km
            s_price = (r["price"] - 1) / 3 if intent.premium else (4 - r["price"]) / 3
            s_rate = max(0.0, (r["rating"] - 3) / 2)                       # 3.0 -> 0, 5.0 -> 1
            s_food = len(hits) / len(intent.foods) if intent.foods else 0.0

            score = (w["distance"] * s_dist + w["price"] * s_price
                     + w["rating"] * s_rate + w["food"] * s_food)
            results.append({
                "name": r["name"], "score": round(score, 3), "distance_km": round(dist, 2),
                "price": r["price"], "rating": r["rating"], "matched": hits,
                "parts": {"distance": round(s_dist, 2), "price": round(s_price, 2),
                          "rating": round(s_rate, 2), "food": round(s_food, 2)},
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------- 3. demo
if __name__ == "__main__":
    DATA = [
        {"name": "Roma Wood Fire", "lat": 28.6330, "lng": 77.2190, "price": 3, "rating": 4.6, "cuisines": ["italian"], "dishes": ["pizza", "pasta"], "veg": False},
        {"name": "Pizza Gali", "lat": 28.5560, "lng": 77.2000, "price": 1, "rating": 4.1, "cuisines": ["italian"], "dishes": ["pizza"], "veg": True},
        {"name": "Nonna's Table", "lat": 28.5270, "lng": 77.2190, "price": 4, "rating": 4.8, "cuisines": ["italian"], "dishes": ["pizza", "pasta"], "veg": False},
        {"name": "Crust & Co.", "lat": 28.5140, "lng": 77.2060, "price": 2, "rating": 4.6, "cuisines": ["italian"], "dishes": ["pizza"], "veg": False},
        {"name": "Biryani Baithak", "lat": 28.6510, "lng": 77.2310, "price": 2, "rating": 4.5, "cuisines": ["mughlai"], "dishes": ["biryani"], "veg": False},
        {"name": "Udupi Sagar", "lat": 28.5530, "lng": 77.2570, "price": 1, "rating": 4.4, "cuisines": ["south indian"], "dishes": ["dosa"], "veg": True},
    ]
    rec = Recommender(DATA)
    me = (28.6315, 77.2167)
    for q in ["less price", "distance", "good pizza", "good italian", "cheap veg"]:
        print(f"\n> {q}")
        for x in rec.recommend(q, *me, top_k=3):
            print(f"  {x['name']:<16} score={x['score']:<5} {x['distance_km']:>5} km  "
                  f"Rs{x['price']}  {x['rating']}*  matched={x['matched']}")

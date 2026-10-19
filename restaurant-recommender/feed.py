"""
Feed-style restaurant recommender (pure Python, no dependencies).
Uses parse_query / haversine_km from recommender.py.

Flow:
  1. New user, no preferences      -> feed is RANDOM (cold start)
  2. User sets preferences, or
     likes / skips a few cards     -> feed becomes PERSONALISED
  3. Even then, ~20% of cards stay random (exploration) so the feed doesn't get stuck

    feed = Feed(restaurants)
    user = feed.new_user()
    feed.next_batch(user, lat, lng)                       # random
    feed.set_preferences(user, cuisines=["italian"], max_price=2, veg=False)
    feed.next_batch(user, lat, lng)                       # personalised
    feed.feedback(user, "Pizza Gali", "like")             # learns from this
"""
from __future__ import annotations
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field

from recommender import haversine_km, parse_query


@dataclass
class UserProfile:
    # explicit preferences (what the user set)
    cuisines: set = field(default_factory=set)      # e.g. {"italian", "pizza"}
    max_price: int | None = None                    # 1-4
    veg: bool = False
    radius_km: float = 15.0
    # implicit preferences (learned from behaviour)
    affinity: dict = field(default_factory=lambda: defaultdict(float))   # tag -> score
    interactions: int = 0
    seen: set = field(default_factory=set)          # names already shown in the feed

    def has_preferences(self) -> bool:
        return bool(self.cuisines) or self.max_price is not None or self.veg


class Feed:
    WEIGHTS = {"explicit": 0.35, "learned": 0.25, "distance": 0.20, "rating": 0.15, "price": 0.05}
    EXPLORE_RATE = 0.2          # share of each batch that is random even when personalised
    WARMUP_INTERACTIONS = 3     # likes/skips needed before behaviour alone personalises the feed
    LIKE, SKIP = 1.0, -0.5      # how much one action moves a tag's affinity

    def __init__(self, restaurants: list[dict], seed: int | None = None):
        self.restaurants = restaurants
        self.by_name = {r["name"]: r for r in restaurants}
        self.rng = random.Random(seed)

    # ------------------------------------------------------------ user + preferences
    def new_user(self) -> UserProfile:
        return UserProfile()

    def set_preferences(self, user: UserProfile, cuisines=None, max_price=None,
                        veg=None, radius_km=None) -> None:
        if cuisines is not None:
            user.cuisines = {c.lower() for c in cuisines}
        if max_price is not None:
            user.max_price = max_price
        if veg is not None:
            user.veg = veg
        if radius_km is not None:
            user.radius_km = radius_km
        user.seen.clear()        # fresh feed after the user changes settings

    def set_preferences_from_text(self, user: UserProfile, text: str) -> None:
        """Reuse the query parser: 'cheap veg italian' -> preferences."""
        it = parse_query(text)
        self.set_preferences(user, cuisines=it.foods or None,
                             max_price=2 if it.cheap else None,
                             veg=True if it.veg else None)

    # ------------------------------------------------------------ learning from behaviour
    def feedback(self, user: UserProfile, name: str, action: str) -> None:
        """action: 'like' or 'skip'. Moves the affinity of every tag on that restaurant."""
        r = self.by_name[name]
        delta = self.LIKE if action == "like" else self.SKIP
        for tag in set(r["cuisines"]) | set(r["dishes"]):
            user.affinity[tag] += delta
        user.interactions += 1

    # ------------------------------------------------------------ the feed
    def phase(self, user: UserProfile) -> str:
        if user.has_preferences() or user.interactions >= self.WARMUP_INTERACTIONS:
            return "personalised"
        return "random"

    def next_batch(self, user: UserProfile, lat: float, lng: float, n: int = 5) -> list[dict]:
        pool = self._eligible(user, lat, lng)
        fresh = [p for p in pool if p["r"]["name"] not in user.seen]
        if len(fresh) < n:                       # ran out of new cards -> start over
            user.seen.clear()
            fresh = pool
        phase = self.phase(user)

        if phase == "random":
            picks = self.rng.sample(fresh, min(n, len(fresh)))
        else:
            for p in fresh:
                p["score"] = self._score(user, p)
            fresh.sort(key=lambda p: p["score"], reverse=True)
            n_explore = max(1, round(n * self.EXPLORE_RATE)) if len(fresh) > n else 0
            top = fresh[: n - n_explore]
            rest = fresh[n - n_explore:]
            picks = top + self.rng.sample(rest, min(n_explore, len(rest)))   # best first, random last

        out = []
        for p in picks:
            user.seen.add(p["r"]["name"])
            out.append({"name": p["r"]["name"], "distance_km": round(p["dist"], 2),
                        "price": p["r"]["price"], "rating": p["r"]["rating"],
                        "cuisines": p["r"]["cuisines"], "phase": phase,
                        "score": round(p.get("score", 0.0), 3)})
        return out

    # ------------------------------------------------------------ internals
    def _eligible(self, user, lat, lng):
        pool = []
        for r in self.restaurants:
            d = haversine_km(lat, lng, r["lat"], r["lng"])
            if d > user.radius_km:
                continue
            if user.veg and not r.get("veg", False):
                continue
            if user.max_price is not None and r["price"] > user.max_price:
                continue
            pool.append({"r": r, "dist": d})
        return pool

    def _score(self, user, p) -> float:
        r, d = p["r"], p["dist"]
        tags = set(r["cuisines"]) | set(r["dishes"])
        w = self.WEIGHTS

        explicit = 1.0 if (user.cuisines & tags) else (0.0 if user.cuisines else 0.5)
        raw = sum(user.affinity.get(t, 0.0) for t in tags) / max(len(tags), 1)
        learned = (math.tanh(raw) + 1) / 2                      # squash to 0..1, 0.5 = neutral
        dist = math.exp(-d / 4.0)
        rating = max(0.0, (r["rating"] - 3) / 2)
        price = (4 - r["price"]) / 3

        return (w["explicit"] * explicit + w["learned"] * learned + w["distance"] * dist
                + w["rating"] * rating + w["price"] * price)


# ---------------------------------------------------------------- demo
if __name__ == "__main__":
    DATA = [
        {"name": "Roma Wood Fire", "lat": 28.633, "lng": 77.219, "price": 3, "rating": 4.6, "cuisines": ["italian"], "dishes": ["pizza", "pasta"], "veg": False},
        {"name": "Pizza Gali", "lat": 28.556, "lng": 77.200, "price": 1, "rating": 4.1, "cuisines": ["italian"], "dishes": ["pizza"], "veg": True},
        {"name": "Pasta La Vista", "lat": 28.560, "lng": 77.175, "price": 2, "rating": 4.3, "cuisines": ["italian"], "dishes": ["pasta", "pizza"], "veg": True},
        {"name": "Biryani Baithak", "lat": 28.651, "lng": 77.231, "price": 2, "rating": 4.5, "cuisines": ["mughlai"], "dishes": ["biryani", "kebab"], "veg": False},
        {"name": "Dilli Dhaba", "lat": 28.544, "lng": 77.233, "price": 1, "rating": 4.3, "cuisines": ["north indian"], "dishes": ["butter chicken", "naan"], "veg": False},
        {"name": "Momo Mela", "lat": 28.582, "lng": 77.205, "price": 1, "rating": 4.0, "cuisines": ["tibetan"], "dishes": ["momos"], "veg": False},
        {"name": "Udupi Sagar", "lat": 28.553, "lng": 77.257, "price": 1, "rating": 4.4, "cuisines": ["south indian"], "dishes": ["dosa", "idli"], "veg": True},
        {"name": "Dragon Pearl", "lat": 28.571, "lng": 77.243, "price": 3, "rating": 4.4, "cuisines": ["chinese"], "dishes": ["noodles"], "veg": False},
        {"name": "Burger Barn", "lat": 28.599, "lng": 77.233, "price": 2, "rating": 3.9, "cuisines": ["american"], "dishes": ["burger"], "veg": False},
        {"name": "Kulfi Kingdom", "lat": 28.608, "lng": 77.240, "price": 1, "rating": 4.7, "cuisines": ["dessert"], "dishes": ["kulfi"], "veg": True},
    ]
    feed = Feed(DATA, seed=7)
    user = feed.new_user()
    me = (28.6315, 77.2167)

    def show(title, batch):
        print(f"\n{title}")
        for c in batch:
            print(f"  [{c['phase']:<12}] {c['name']:<16} {c['distance_km']:>5} km  Rs{c['price']}  {c['rating']}*  {c['cuisines']}")

    show("1) New user, nothing set -> random feed", feed.next_batch(user, *me, n=4))

    feed.set_preferences(user, cuisines=["italian"], max_price=2)
    show("2) User sets: Italian, budget up to Rs Rs -> personalised feed", feed.next_batch(user, *me, n=4))

    feed.feedback(user, "Pizza Gali", "like")
    feed.feedback(user, "Burger Barn", "skip")
    feed.feedback(user, "Pasta La Vista", "like")
    user.seen.clear()   # simulate a feed refresh so already-shown cards can rank again
    show("3) After likes/skips (feed refreshed) -> ranking adapts", feed.next_batch(user, *me, n=4))
    print("\nLearned affinities:", {k: v for k, v in sorted(user.affinity.items(), key=lambda kv: -kv[1])[:4]})

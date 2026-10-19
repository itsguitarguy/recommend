import json
from recommender import Recommender
from feed import Feed

restaurants = json.load(open("data/restaurants.json", encoding="utf-8"))
lat, lng = 28.6315, 77.2167          # user's location (replace with real GPS)

# 1) Search by text
rec = Recommender(restaurants)
print("Search: 'cheap good pizza nearby'")
for x in rec.recommend("cheap good pizza nearby", lat, lng, top_k=3):
    print(f"  {x['name']:<16} {x['distance_km']} km  score={x['score']}")

# 2) Feed: random -> personalised
feed = Feed(restaurants, seed=1)
user = feed.new_user()
print("\nFeed (new user, random):")
for c in feed.next_batch(user, lat, lng, n=3):
    print(f"  {c['name']:<16} {c['phase']}")

feed.set_preferences(user, cuisines=["italian"], max_price=2)
print("\nFeed (after setting preferences):")
for c in feed.next_batch(user, lat, lng, n=3):
    print(f"  {c['name']:<16} {c['phase']}")

feed.feedback(user, "Pizza Gali", "like")

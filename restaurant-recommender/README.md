# Restaurant Recommender

A small, dependency-free, location-aware restaurant recommendation model in Python.

- **Search**: type what you want ("cheap good pizza nearby", "less price", "distance", "good Italian") and get ranked restaurants.
- **Feed**: a feed that starts **random**, then becomes **personalised** from preferences the user sets and from likes/skips.

## How it works

Each restaurant gets four scores from 0 to 1: distance (haversine from user to the restaurant's lat/lng), price, rating and food match. The words in the query set the weights, so "distance" boosts distance and "less price" boosts price. Hard filters apply for radius, veg-only and requested food.

The feed adds a learned per-tag affinity (like +1, skip -0.5), keeps 20% of each batch random for exploration, and never repeats a card until the user has seen everything nearby.

## Quick start

```bash
python main.py            # demo of search + feed
python -m unittest discover tests
```

No installs needed (Python 3.9+).

## Usage

```python
import json
from recommender import Recommender
from feed import Feed

restaurants = json.load(open("data/restaurants.json"))

# search
Recommender(restaurants).recommend("cheap good pizza", lat=28.63, lng=77.21)

# feed
feed = Feed(restaurants)
user = feed.new_user()
feed.next_batch(user, 28.63, 77.21, n=5)                         # random
feed.set_preferences(user, cuisines=["italian"], max_price=2)
feed.next_batch(user, 28.63, 77.21, n=5)                         # personalised
feed.feedback(user, "Pizza Gali", "like")                        # learns
```

## Data format

Each restaurant in `data/restaurants.json`:

| field | type | notes |
|---|---|---|
| name | str | unique |
| lat, lng | float | restaurant geo-location |
| price | int | 1 (cheap) to 4 (expensive) |
| rating | float | 0 to 5 |
| cuisines, dishes | list[str] | lowercase tags |
| veg | bool | pure veg or not |

The sample restaurants are made up. Replace them with your own data.

## Roadmap

- Geotagged photo (EXIF) input and image-similarity recommendations
- Save user profiles in a database
- Learn the scoring weights from real click data
- Web API (FastAPI)

## License

MIT

import json, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from recommender import Recommender, parse_query, haversine_km
from feed import Feed

DATA = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "restaurants.json"), encoding="utf-8"))
ME = (28.6315, 77.2167)


class Tests(unittest.TestCase):
    def test_haversine_zero(self):
        self.assertAlmostEqual(haversine_km(*ME, *ME), 0.0)

    def test_parse_query(self):
        it = parse_query("cheap good pizza nearby")
        self.assertEqual(it.foods, ["pizza"])
        self.assertTrue(it.cheap and it.good and it.near)

    def test_food_filter(self):
        for x in Recommender(DATA).recommend("good italian", *ME):
            self.assertIn("italian", [r for r in DATA if r["name"] == x["name"]][0]["cuisines"])

    def test_cheap_first(self):
        top = Recommender(DATA).recommend("less price", *ME, top_k=1)[0]
        self.assertEqual(top["price"], 1)

    def test_feed_phases(self):
        feed = Feed(DATA, seed=1); u = feed.new_user()
        self.assertEqual(feed.next_batch(u, *ME, n=3)[0]["phase"], "random")
        feed.set_preferences(u, cuisines=["italian"])
        self.assertEqual(feed.next_batch(u, *ME, n=3)[0]["phase"], "personalised")

    def test_no_repeats_in_feed(self):
        feed = Feed(DATA, seed=1); u = feed.new_user()
        a = {c["name"] for c in feed.next_batch(u, *ME, n=4)}
        b = {c["name"] for c in feed.next_batch(u, *ME, n=4)}
        self.assertFalse(a & b)

    def test_like_updates_affinity(self):
        feed = Feed(DATA); u = feed.new_user()
        feed.feedback(u, "Pizza Gali", "like")
        self.assertGreater(u.affinity["pizza"], 0)


if __name__ == "__main__":
    unittest.main()

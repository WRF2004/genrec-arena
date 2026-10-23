"""Convert user-downloaded MovieLens 1M (implicit positive ratings >=4)."""
import argparse
import csv
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--raw", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
with (a.output / "items.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["item_id", "text"])
    for line in (a.raw / "movies.dat").read_text(encoding="latin-1").splitlines():
        item, title, genres = line.split("::")
        w.writerow([item, title + " " + genres.replace("|", " ")])
with (a.output / "events.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["user_id", "item_id", "timestamp"])
    for line in (a.raw / "ratings.dat").read_text().splitlines():
        user, item, rating, stamp = line.split("::")
        if float(rating) >= 4:
            w.writerow([user, item, stamp])
print("Converted positive feedback only. Choose global chronological cutoffs before training.")

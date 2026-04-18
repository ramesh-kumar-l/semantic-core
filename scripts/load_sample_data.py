"""
Load 80 sample items into the running service and run test queries.
Usage:
    python scripts/load_sample_data.py [--url http://localhost:8000]
"""
import argparse
import time
import httpx

SAMPLE_TEXTS = [
    # nature / outdoors
    "Hiking through mountain trails with breathtaking views",
    "Sunset over a snow-capped mountain range",
    "A quiet lake surrounded by pine forests",
    "Rocky cliffs overlooking the ocean",
    "Rolling green hills covered in wildflowers",
    "A dense rainforest with exotic wildlife",
    "Waterfall cascading into a turquoise pool",
    "Desert sand dunes under a blazing sun",
    "Arctic tundra and the Northern Lights",
    "Volcanic landscape with lava flows",
    # beach / ocean
    "Sandy beach with crystal-clear tropical water",
    "Surfing massive waves at a famous beach",
    "Snorkeling among colorful coral reefs",
    "Fishing boat returning to a coastal village",
    "Dolphins jumping near a sailing yacht",
    "Collecting shells along a deserted shoreline",
    "Parasailing over a turquoise lagoon",
    "Beach bonfire under a starry sky",
    "Kayaking through sea caves",
    "Mangrove swamps along a tropical coast",
    # food / cuisine
    "Homemade pasta with fresh tomato sauce",
    "Grilling steaks over an open fire",
    "Sushi platter with fresh sashimi",
    "Spicy Thai curry with jasmine rice",
    "French croissants fresh from the oven",
    "Street tacos with salsa verde",
    "Indian biryani slow-cooked with spices",
    "Chocolate lava cake for dessert",
    "Farmers market with seasonal vegetables",
    "Wood-fired Neapolitan pizza",
    # travel / cities
    "Exploring ancient ruins in Rome",
    "Night markets in Bangkok",
    "Tokyo skyline at night",
    "Street art in Berlin",
    "Gondola ride through Venice canals",
    "Watching the sunrise from the Eiffel Tower",
    "Safari in the Serengeti",
    "Road trip across the American Southwest",
    "Train journey through the Swiss Alps",
    "Cycling through Amsterdam's city centre",
    # technology
    "Machine learning model training on GPU clusters",
    "Building a REST API with FastAPI and Python",
    "Deploying containers with Kubernetes",
    "Real-time data streaming with Apache Kafka",
    "Vector search with semantic embeddings",
    "Graph neural networks for recommendation systems",
    "Quantum computing breakthroughs in 2024",
    "Self-driving cars navigating urban environments",
    "Augmented reality apps for mobile devices",
    "Open-source large language models",
    # sports
    "Marathon runner crossing the finish line",
    "Basketball slam dunk in overtime",
    "Mountain biking on extreme downhill trails",
    "Rock climbing a sheer granite face",
    "Swimming across an open-water channel",
    "Chess grandmaster tournament finals",
    "Formula 1 race car on a wet track",
    "Olympic gymnastics floor routine",
    "Freestyle ski jumping in fresh powder",
    "Soccer match in a packed stadium",
    # arts / culture
    "Jazz musician improvising at a late-night club",
    "Oil painting of a stormy seascape",
    "Ballet performance at an opera house",
    "Sculpture garden in a modern art museum",
    "Indie film festival screenings",
    "Poetry reading in a dimly lit bookstore",
    "Photography exhibition on urban decay",
    "Classical orchestra performing Beethoven",
    "Street musician playing flamenco guitar",
    "Traditional Japanese tea ceremony",
    # science
    "Black hole imaged by a radio telescope array",
    "Deep sea creatures discovered near hydrothermal vents",
    "CRISPR gene editing curing genetic diseases",
    "Mars rover discovering ancient riverbeds",
    "Gravitational wave detection by LIGO",
    "Coral reef bleaching caused by climate change",
    "Renewable energy from offshore wind farms",
    "Electric vehicle battery recycling research",
    "Reforestation drone planting thousands of trees",
    "Urban vertical farms growing food efficiently",
]

TEST_QUERIES = [
    ("beach", 5),
    ("food", 5),
    ("mountain", 5),
    ("technology AI", 5),
    ("ocean water", 5),
]


def main(base_url: str) -> None:
    client = httpx.Client(base_url=base_url, timeout=30)

    print(f"Ingesting {len(SAMPLE_TEXTS)} items into {base_url} …")
    ids = []
    t_start = time.perf_counter()
    for text in SAMPLE_TEXTS:
        r = client.post("/content", json={"text": text})
        r.raise_for_status()
        ids.append(r.json()["content_id"])
    elapsed = time.perf_counter() - t_start
    print(f"  Done. {len(ids)} items in {elapsed:.2f}s ({elapsed/len(ids)*1000:.1f}ms/item)\n")

    print("Running test queries:")
    for query, k in TEST_QUERIES:
        t0 = time.perf_counter()
        r = client.post("/similar", json={"query": query, "k": k})
        r.raise_for_status()
        ms = (time.perf_counter() - t0) * 1000
        results = r.json()["results"]
        print(f"  query={query!r:20s}  latency={ms:.1f}ms")
        for res in results:
            # Find original text for display
            idx = ids.index(res["id"]) if res["id"] in ids else -1
            text_preview = SAMPLE_TEXTS[idx][:60] if idx >= 0 else "?"
            print(f"    score={res['score']:.4f}  {text_preview}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    main(args.url)

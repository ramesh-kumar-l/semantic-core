"""
Load the three "golden example" datasets into a running Semantic Core Service.

Each example is seeded into TWO subsystems so it is fully visible the moment the
application launches:

  1. A searchable content namespace  (POST /content)   -> vector + BM25 retrieval
  2. The semantic graph              (POST /semantic/ingest) -> nodes + auto-linked
                                                               person / module / milestone
                                                               entities and edges

The three examples are deliberately about *engineering work*, so the platform is
shown indexing real software-engineering knowledge:

  - semantic_core   : building this very platform (Semantic Core Service)
  - netflix_clone   : building a Netflix-style streaming platform
  - fitness_tracker : building an Android fitness-tracking app

Usage:
    # 1. start the API (defaults are fine: memory + graph enabled)
    uvicorn app.main:app --reload
    # 2. run this script
    python scripts/load_golden_examples.py [--url http://localhost:8000] [--reset]

The metadata convention used for every document node:

    people   -> the engineering role that owned the work   (e.g. "MLEngineer")
    location -> the module / subsystem it belongs to        (e.g. "hybrid_search")
    event    -> the milestone / epic it was delivered under  (e.g. "E3 Hybrid Retrieval")

These map onto the graph linker, which creates `contains` (-> person),
`occurs_at` (-> location) and `happens_during` (-> event) edges automatically,
giving you a navigable who / what / when graph per example.
"""
import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List

import httpx


# --------------------------------------------------------------------------- #
# Data model                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class Doc:
    id: str            # stable graph-node id, e.g. "scs-03"
    title: str         # short human label
    role: str          # -> graph person node
    module: str        # -> graph location node
    milestone: str     # -> graph event node
    text: str          # full searchable narrative


@dataclass
class Example:
    namespace: str
    title: str
    summary: str
    docs: List[Doc] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Example 1 — Building the Semantic Core Service itself                        #
# --------------------------------------------------------------------------- #

SEMANTIC_CORE = Example(
    namespace="semantic_core",
    title="Building the Semantic Core Service",
    summary="The engineering journey of this retrieval platform: layered design, "
            "hybrid search, feedback learning, and a semantic graph.",
    docs=[
        Doc("scs-01", "Why a layered retrieval platform, not a toy vector demo",
            "PlatformEngineer", "architecture", "E0 Foundations",
            "Most semantic-search projects start as a single embed-and-cosine script and "
            "collapse under real requirements. Semantic Core Service is designed as four "
            "stacked layers behind one FastAPI app: vector retrieval, hybrid retrieval, "
            "feedback-driven ranking, and a semantic graph. Every layer is feature-flagged "
            "so the system runs with zero config locally yet can scale toward a production "
            "search stack without rewrites. The guiding principle is stable interfaces with "
            "swappable implementations."),
        Doc("scs-02", "Namespaced memory for multi-tenant isolation",
            "PlatformEngineer", "memory_namespaces", "E1 Namespaced Memory",
            "Retrieval data is partitioned by namespace so tenants, domains, or experiments "
            "never bleed into each other. Each namespace owns its own vector index and BM25 "
            "index, lazily hydrated from disk on first access and persisted back under "
            "data/<namespace>/. Lazy loading keeps cold-start cost proportional to what you "
            "actually query, and disk persistence means restarts do not lose ingested content."),
        Doc("scs-03", "A stable VectorStore interface with pluggable backends",
            "BackendEngineer", "vector_store", "E1 Namespaced Memory",
            "Business logic depends only on a tiny VectorStore contract: add(id, vector, text) "
            "and search(vector, k). FlatIndex implements an in-memory cosine scan for fast local "
            "iteration; QdrantStore wraps a production HNSW vector database with the same "
            "interface. The API layer never imports Qdrant directly, so swapping backends is a "
            "factory change, not a refactor. This is dependency inversion applied to retrieval."),
        Doc("scs-04", "Hybrid retrieval: fusing BM25 and vector search",
            "MLEngineer", "hybrid_search", "E3 Hybrid Retrieval",
            "Pure vector search misses exact keyword matches; pure BM25 misses paraphrases. "
            "Hybrid retrieval runs both and fuses them with a weighted, score-normalized blend "
            "controlled by alpha. Vector candidates capture semantic similarity while BM25 "
            "candidates capture lexical precision, and fusion recovers results that either method "
            "alone would rank too low. Alpha lets operators tune the recall/precision balance per "
            "workload."),
        Doc("scs-05", "Score normalization and the pitfalls of naive fusion",
            "MLEngineer", "hybrid_search", "E3 Hybrid Retrieval",
            "Cosine scores and BM25 scores live on different scales, so adding them directly lets "
            "one signal dominate. The fusion layer normalizes each result set before blending and "
            "emits retrieval warnings when the two methods fully disagree or when the score spread "
            "is suspiciously large. These diagnostics turn silent relevance bugs into observable, "
            "log-greppable events."),
        Doc("scs-06", "Reranking: deterministic first, learned second",
            "MLEngineer", "learning_to_rank", "E4 Reranking",
            "After fusion, results pass through a reranking stage. A deterministic reranker applies "
            "lightweight lexical and length heuristics for reproducibility, then an optional "
            "learning-to-rank stage reorders by learned signals. The two are layered so the system "
            "always has a sane ordering even before any feedback exists, and improves smoothly as "
            "signals accumulate."),
        Doc("scs-07", "Feedback-aware ranking from clicks (CTR and MRR)",
            "MLEngineer", "learning_to_rank", "E7 Feedback Learning",
            "Every click on a result is recorded as a feedback event with the query, namespace, "
            "result set, clicked id, and position. A rule-based learning-to-rank model converts "
            "these into per-query CTR and MRR signals that boost previously successful results. "
            "Because it is rule-based, it bootstraps from the very first click instead of waiting "
            "for a trained model, and degrades gracefully when no history exists."),
        Doc("scs-08", "An adaptive query planner driven by historical performance",
            "PlatformEngineer", "query_planner", "E9 Adaptive Planner",
            "Different queries deserve different retrieval strategies. The planner chooses alpha, "
            "vector_k, and bm25_k per query. In rule-based mode it uses query shape; in adaptive "
            "mode it consults a per-query store of historical CTR and MRR and shifts the plan toward "
            "whatever produced clicks before. This closes the loop between feedback collection and "
            "retrieval strategy selection."),
        Doc("scs-09", "The semantic graph: metadata extraction and entity linking",
            "PlatformEngineer", "semantic_graph", "E8 Semantic Graph",
            "Beyond ranked lists, the platform builds a graph. Ingested objects are run through a "
            "metadata extractor and a relation linker that materialize person, location, time, and "
            "event nodes and connect the object to them with typed edges. A linking engine then "
            "scores nearby nodes and auto-creates related_to edges above a threshold, so the graph "
            "grows denser as more content arrives."),
        Doc("scs-10", "Bounded graph traversal and the graph query engine",
            "BackendEngineer", "semantic_graph", "E8 Semantic Graph",
            "Graph queries resolve anchors (a person, location, or event) and walk inbound edges to "
            "find the objects that reference them, intersecting across multiple filters. All "
            "traversal is depth-bounded and emits traversal diagnostics, preventing runaway BFS over "
            "a dense graph. When no anchor resolves, the engine returns an empty set so callers can "
            "fall back to vector retrieval instead of erroring."),
        Doc("scs-11", "Observability: structured logs, request ids, and retrieval warnings",
            "BackendEngineer", "observability", "Observability",
            "Every request carries a request id and emits structured JSON logs with per-stage "
            "latency: embedding_ms, store_ms, search_ms. The retrieval path proactively warns on "
            "empty results, low top score, high latency, empty fusion, and large score divergence. "
            "These signals make relevance regressions debuggable from logs alone, without a "
            "reproduction harness."),
        Doc("scs-12", "Graceful degradation as a first-class design rule",
            "PlatformEngineer", "architecture", "E0 Foundations",
            "Optional dependencies never hard-fail the service. Missing sentence-transformers falls "
            "back to deterministic hash embeddings; an unavailable Neo4j falls back to the SQLite "
            "graph; a missing ML reranker falls back to the rule-based reranker; a failing LLM "
            "rewrite falls back to the original query. The service therefore starts and serves with "
            "zero external infrastructure, which is what makes it genuinely local-first."),
        Doc("scs-13", "Query intelligence: analysis, rewrite, and multi-query expansion",
            "MLEngineer", "query_intelligence", "Intelligence",
            "An optional intelligence layer analyzes the query, classifies its type, and rewrites it "
            "for better recall. In LLM mode it can expand one query into several paraphrases and fuse "
            "their results by max score. Every LLM call is timeout-bounded and falls back to the "
            "original query, so intelligence improves results when available but is never on the "
            "critical failure path."),
    ],
)


# --------------------------------------------------------------------------- #
# Example 2 — Building a Netflix-style streaming platform                      #
# --------------------------------------------------------------------------- #

NETFLIX_CLONE = Example(
    namespace="netflix_clone",
    title="Building a Netflix-style streaming platform",
    summary="System design for a streaming service — catalog, adaptive playback, CDN, "
            "search and recommendations — with Semantic Core powering discovery.",
    docs=[
        Doc("nfx-01", "System overview: a service-oriented streaming platform",
            "StreamingEngineer", "architecture", "M1 Catalog And Metadata",
            "The platform is decomposed into independently deployable services: catalog, playback, "
            "search, recommendations, profiles, and billing. Each owns its data and exposes a narrow "
            "API. A gateway handles auth and routing. This separation lets the recommendations team "
            "iterate on models without touching the playback path, and lets playback scale on QoE "
            "metrics independently of catalog write traffic."),
        Doc("nfx-02", "Modeling the content catalog and metadata",
            "BackendEngineer", "catalog_service", "M1 Catalog And Metadata",
            "Titles, seasons, episodes, cast, genres, and localized artwork are modeled as a graph of "
            "related entities rather than flat rows. Metadata feeds both search and recommendations, "
            "so it is treated as a first-class product surface with validation, versioning, and an "
            "ingestion pipeline that normalizes data from multiple content providers."),
        Doc("nfx-03", "Adaptive bitrate playback with an ABR ladder",
            "StreamingEngineer", "playback_engine", "M2 Adaptive Playback",
            "Video is transcoded into an ABR ladder of resolutions and bitrates and segmented for "
            "HLS and DASH. The client player measures throughput and buffer health and switches rungs "
            "to maximize quality without rebuffering. Per-title encoding tunes the ladder so simple "
            "content does not waste bits and complex content keeps detail."),
        Doc("nfx-04", "CDN and edge caching strategy",
            "StreamingEngineer", "cdn_delivery", "M5 CDN Delivery",
            "Segments are served from edge caches close to viewers, with origin shielding to protect "
            "the storage tier. Popular titles are pre-warmed to the edge; long-tail content is pulled "
            "on demand. Cache keys include the ABR rung so each quality level is cached independently, "
            "and signed URLs enforce entitlement at the edge."),
        Doc("nfx-05", "Catalog search powered by Semantic Core Service",
            "BackendEngineer", "search", "M3 Search",
            "Title search is delivered by embedding synopses and titles into Semantic Core Service and "
            "running hybrid retrieval. Vector search matches a vague mood query like 'mind-bending space "
            "thriller' while BM25 nails exact title lookups, and fusion serves both well. Click feedback "
            "from the search box feeds the learning-to-rank layer so popular results rise over time."),
        Doc("nfx-06", "Recommendations: candidate generation then ranking",
            "MLEngineer", "recommendations", "M4 Recommendations",
            "Recommendations use a two-stage funnel. Candidate generation retrieves a few hundred "
            "plausible titles per user from embedding similarity and co-watch signals; a ranking model "
            "then orders them by predicted watch probability given context. Decoupling the stages keeps "
            "latency low while allowing a richer, slower ranker on a small candidate set."),
        Doc("nfx-07", "Personalized profiles and viewing history",
            "BackendEngineer", "profiles", "M6 Profiles",
            "Each account holds multiple profiles, each with its own taste vector built from viewing "
            "history, ratings, and explicit thumbs. Profile separation keeps a child's cartoons from "
            "polluting an adult's recommendations. History is stored as an append-only event log so "
            "models can be recomputed and audited."),
        Doc("nfx-08", "Cross-device watch state and resume",
            "BackendEngineer", "playback_engine", "M2 Adaptive Playback",
            "Playback position is checkpointed periodically and on pause, keyed by profile and title, so "
            "a viewer can resume on any device. The watch-state service is write-heavy and eventually "
            "consistent: a slightly stale resume point is acceptable, but losing progress is not, so "
            "writes are durable and reads tolerate lag."),
        Doc("nfx-09", "Subscription billing and entitlement tiers",
            "BackendEngineer", "billing", "M6 Profiles",
            "Plans map to entitlements — concurrent streams, max resolution, and download rights — that "
            "the playback and CDN layers enforce. Billing runs on an idempotent event pipeline so retried "
            "webhooks never double-charge, and entitlement changes propagate to active sessions without "
            "forcing a re-login."),
        Doc("nfx-10", "Experimentation and the discovery feedback loop",
            "MLEngineer", "recommendations", "M4 Recommendations",
            "Every row on the home page is an experiment. Impressions and plays are logged and fed back as "
            "ranking signals, mirroring the click-feedback loop in Semantic Core Service: surface results, "
            "watch what users engage with, and let that reshape future ordering. A/B tests gate every "
            "model change behind a measured lift in engagement."),
        Doc("nfx-11", "Streaming quality of experience (QoE) observability",
            "StreamingEngineer", "observability", "M5 CDN Delivery",
            "QoE is measured continuously: startup time, rebuffer ratio, average bitrate, and play "
            "failures, sliced by device, region, and CDN. Alerts fire when rebuffer ratio crosses a "
            "threshold in any region, because a CDN regression in one geography is invisible in a global "
            "average. The same observability discipline applies to search latency and error rates."),
        Doc("nfx-12", "The content ingestion pipeline",
            "BackendEngineer", "catalog_service", "M1 Catalog And Metadata",
            "New titles flow through an ingestion pipeline: validate source metadata, transcode the ABR "
            "ladder, generate thumbnails and trick-play sprites, embed synopses for search and "
            "recommendations, and publish atomically so a title never appears half-ingested. Each stage is "
            "idempotent and resumable so a failure mid-pipeline does not corrupt the catalog."),
    ],
)


# --------------------------------------------------------------------------- #
# Example 3 — Building an Android fitness-tracking app                         #
# --------------------------------------------------------------------------- #

FITNESS_TRACKER = Example(
    namespace="fitness_tracker",
    title="Building an Android fitness-tracking app",
    summary="Engineering an offline-first Android fitness tracker — sensors, sync, "
            "on-device ML, and Compose UI — with Semantic Core for workout search.",
    docs=[
        Doc("fit-01", "App architecture: MVVM and Clean Architecture on Jetpack",
            "MobileEngineer", "app_architecture", "F1 Foundations",
            "The app is layered into presentation (Jetpack Compose + ViewModels), domain (use cases and "
            "models), and data (repositories over Room and remote APIs). Unidirectional data flow with "
            "Kotlin Flow makes state predictable, and the strict layer boundaries let the team unit-test "
            "domain logic without an emulator. Dependency injection wires the layers via Hilt."),
        Doc("fit-02", "The sensor pipeline: accelerometer, GPS, and heart rate",
            "MobileEngineer", "sensors", "F2 Sensor Pipeline",
            "Activity data comes from the accelerometer, fused location, and a heart-rate sensor. A "
            "foreground service keeps sampling alive during a workout while showing a persistent "
            "notification, satisfying Android's background-execution limits. Sensor batching lets the SoC "
            "stay in low-power states between batches, which is the difference between a 40-minute and a "
            "4-hour battery hit on a long run."),
        Doc("fit-03", "Step counting and workout segmentation algorithms",
            "MobileEngineer", "workout_tracking", "F3 Workout Tracking",
            "Steps are derived by peak detection over a low-pass-filtered accelerometer signal, with a "
            "cadence band to reject noise from hand movement. Workouts are segmented into laps and rest "
            "periods using speed and heart-rate thresholds, and GPS traces are simplified with "
            "Douglas-Peucker before storage to keep routes small without losing shape."),
        Doc("fit-04", "Offline-first storage with Room and a WorkManager sync engine",
            "MobileEngineer", "sync_engine", "F4 Offline Sync",
            "All workouts are written to a local Room database first, so the app is fully usable with no "
            "network — essential on a trail run. A WorkManager job syncs pending records when connectivity "
            "returns, with exponential backoff. Each record carries a client-generated id and version so "
            "the backend can resolve conflicts deterministically."),
        Doc("fit-05", "Integrating Health Connect as the system source of truth",
            "MobileEngineer", "health_connect", "F5 Health Connect",
            "The app reads and writes through Android Health Connect so steps, heart rate, and workouts "
            "interoperate with other health apps instead of living in a silo. Granular runtime permissions "
            "let users share exactly the data types they choose, and a sync adapter reconciles Health "
            "Connect records with the app's own database on a schedule."),
        Doc("fit-06", "Building the UI with Jetpack Compose and Material 3",
            "UXEngineer", "ui_compose", "F6 Compose UI",
            "Screens are built in Jetpack Compose with Material 3 theming and dynamic color. State is "
            "hoisted to ViewModels and rendered from immutable UI-state objects, so a workout screen "
            "redraws only what changed. Animated rings and charts are drawn on Compose Canvas, and the "
            "same composables drive a Wear OS companion tile."),
        Doc("fit-07", "On-device ML for activity recognition and workout search",
            "MLEngineer", "ml_insights", "F7 ML Insights",
            "A small TensorFlow Lite model classifies activity type — walk, run, cycle — from accelerometer "
            "windows entirely on-device, so raw motion data never leaves the phone. Past workouts are also "
            "embedded and indexed in Semantic Core Service, letting users ask 'long hilly runs in the rain' "
            "and get semantically matched sessions instead of only date filters."),
        Doc("fit-08", "Battery optimization and surviving Doze",
            "MobileEngineer", "sensors", "F2 Sensor Pipeline",
            "Background work is scheduled through WorkManager so it respects Doze and App Standby buckets "
            "instead of fighting them. Location is requested at the coarsest accuracy that still serves the "
            "feature, sensor sampling is batched, and wake locks are scoped to active workouts only. Battery "
            "Historian profiling guides every tradeoff between data fidelity and drain."),
        Doc("fit-09", "Reminders, goals, and notifications",
            "UXEngineer", "ui_compose", "F6 Compose UI",
            "Daily goals and streaks are nudged through scheduled notifications that respect the user's quiet "
            "hours and notification channels. Goal progress is computed locally so a reminder is accurate even "
            "offline, and deep links jump straight from a notification into the relevant workout or goal "
            "screen."),
        Doc("fit-10", "The backend sync API and conflict resolution",
            "BackendEngineer", "sync_engine", "F4 Offline Sync",
            "The sync API accepts batches of workout records keyed by client id and version. Last-writer-wins "
            "is used for simple fields while structured merges preserve independent edits, and the server "
            "returns a reconciled view the client adopts. Because clients generate ids, retries are idempotent "
            "and a flaky connection never creates duplicate workouts."),
        Doc("fit-11", "Privacy, consent, and data security",
            "BackendEngineer", "privacy_security", "F8 Privacy",
            "Health data is sensitive, so the on-device database is encrypted, network calls are TLS-pinned, "
            "and the app collects only what a feature needs. Consent is explicit and revocable per data type, "
            "and a clear export-and-delete flow lets users take their data or erase it, aligning with health-"
            "data regulations."),
        Doc("fit-12", "Testing strategy: unit, instrumented, and UI tests",
            "MobileEngineer", "app_architecture", "F1 Foundations",
            "Domain logic is covered by fast JVM unit tests; repositories are verified against an in-memory "
            "Room database; and critical flows — start workout, lose network, resync — are exercised with "
            "instrumented Espresso and Compose UI tests on emulators in CI. Sensor input is faked through a "
            "test double so workout logic is reproducible without a treadmill."),
    ],
)


EXAMPLES: List[Example] = [SEMANTIC_CORE, NETFLIX_CLONE, FITNESS_TRACKER]


# Demo queries run after seeding to prove the data is live and viewable.
DEMO_SEARCHES = [
    ("semantic_core", "how does hybrid search combine keyword and vector"),
    ("semantic_core", "learning from user clicks to improve ranking"),
    ("netflix_clone", "adaptive video streaming quality"),
    ("netflix_clone", "how are recommendations generated"),
    ("fitness_tracker", "tracking workouts without internet"),
    ("fitness_tracker", "on device machine learning for activity"),
]


# --------------------------------------------------------------------------- #
# Seeding                                                                      #
# --------------------------------------------------------------------------- #

def _delete_namespace_docs(client: httpx.Client, example: Example) -> None:
    """Best-effort removal of previously seeded graph nodes (content is overwritten)."""
    for doc in example.docs:
        try:
            client.delete(f"/content/{doc.id}", params={"namespace": example.namespace})
        except httpx.HTTPError:
            pass


def seed_example(client: httpx.Client, example: Example) -> Dict[str, int]:
    content_ok = 0
    graph_ok = 0
    links = 0

    for doc in example.docs:
        # 1) searchable content document (vector + BM25), persisted to data/<ns>/
        r = client.post("/content", json={
            "text": doc.text,
            "namespace": example.namespace,
            "metadata": {
                "doc_id": doc.id,
                "title": doc.title,
                "role": doc.role,
                "module": doc.module,
                "milestone": doc.milestone,
            },
        })
        r.raise_for_status()
        content_ok += 1

        # 2) semantic graph node + auto-linked person / module / milestone entities
        r = client.post("/semantic/ingest", json={
            "id": doc.id,
            "type": "document",
            "text": doc.title,
            "metadata": {
                "title": doc.title,
                "summary": doc.text[:140],
                "project": example.namespace,
                "people": [doc.role],
                "location": doc.module,
                "event": doc.milestone,
            },
        })
        r.raise_for_status()
        body = r.json()
        graph_ok += 1
        # links_created is logged server-side; not returned, so we just count nodes.

    return {"content": content_ok, "graph": graph_ok, "links": links}


def run_demos(client: httpx.Client) -> None:
    print("\nDemo searches (proof the data is live):")
    for namespace, query in DEMO_SEARCHES:
        t0 = time.perf_counter()
        r = client.post("/similar", json={"query": query, "k": 3, "namespace": namespace})
        r.raise_for_status()
        ms = (time.perf_counter() - t0) * 1000
        results = r.json()["results"]
        top = results[0]["score"] if results else float("nan")
        print(f"  [{namespace:15s}] {query!r}")
        print(f"      hits={len(results)} top_score={top:.4f} latency={ms:.1f}ms")

    print("\nDemo semantic graph queries:")
    graph_demos = [
        {"type": "document", "label": "all document nodes"},
        {"event": "E3 Hybrid Retrieval", "label": "docs under milestone 'E3 Hybrid Retrieval'"},
        {"person": "MLEngineer", "label": "everything owned by MLEngineer"},
        {"location": "playback_engine", "label": "docs in module 'playback_engine'"},
    ]
    for demo in graph_demos:
        label = demo.pop("label")
        r = client.post("/semantic/query", json={**demo, "k": 25})
        r.raise_for_status()
        total = r.json()["total"]
        print(f"  {label:48s} -> {total} node(s)")


def main(base_url: str, reset: bool) -> None:
    client = httpx.Client(base_url=base_url, timeout=60)

    # Confirm the service is reachable and graph is enabled.
    try:
        health = client.get("/admin/health").json()
    except httpx.HTTPError:
        raise SystemExit(f"Service not reachable at {base_url}. Start it with: "
                         f"uvicorn app.main:app --reload")
    if not health.get("feature_flags", {}).get("graph_enabled", False):
        print("WARNING: graph is disabled (GRAPH_ENABLED=false); graph nodes will be skipped "
              "by the server. Restart with graph enabled for the full demo.")

    total = {"content": 0, "graph": 0}
    for example in EXAMPLES:
        if reset:
            _delete_namespace_docs(client, example)
        print(f"\nSeeding '{example.namespace}' — {example.title}")
        print(f"  {len(example.docs)} documents …")
        t0 = time.perf_counter()
        counts = seed_example(client, example)
        dt = time.perf_counter() - t0
        total["content"] += counts["content"]
        total["graph"] += counts["graph"]
        print(f"  done: {counts['content']} searchable docs + {counts['graph']} graph nodes "
              f"in {dt:.1f}s")

    run_demos(client)

    health = client.get("/admin/health").json()
    print("\n/admin/health after seeding:")
    print(f"  namespaces     : {health.get('namespaces')}")
    print(f"  doc_counts     : {health.get('doc_counts')}")
    print(f"  graph_node_cnt : {health.get('graph_node_count')}")
    print(f"\nTotals: {total['content']} searchable documents, "
          f"{total['graph']} graph document nodes across {len(EXAMPLES)} examples.")
    print("\nOpen the Streamlit UI (streamlit run streamlit_app.py) and try the "
          "Search / Semantic Query / System Health tabs with the namespaces above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--reset", action="store_true",
                        help="delete previously seeded graph nodes before re-seeding")
    args = parser.parse_args()
    main(args.url, args.reset)

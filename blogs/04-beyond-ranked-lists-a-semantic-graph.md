---
title: "Beyond Ranked Lists: Building a Semantic Graph for Retrieval"
subtitle: "Some questions aren't ranking problems. 'Show me everything this engineer touched' is a graph traversal — here's how to build one that stays bounded."
tags: [Knowledge Graph, Semantic Search, Graph Databases, Information Retrieval, System Design]
series: "Semantic Core Service"
part: 4
---

# Beyond Ranked Lists: Building a Semantic Graph for Retrieval

The first three posts in this series built a strong ranked-list engine:
[vectors](01-why-a-retrieval-platform-not-a-vector-demo.md),
[hybrid retrieval](02-hybrid-retrieval-bm25-and-vectors.md), and
[learned ranking](03-teaching-search-to-learn-feedback-and-l2r.md). For "find me documents that
match this query," that stack is hard to beat.

But some questions are not ranking questions at all. Consider:

- *"Show me everything the ML engineer worked on across the whole project."*
- *"Which documents belong to the **E3 Hybrid Retrieval** milestone?"*
- *"Starting from this design decision, what else is connected to it, and how?"*

You can hammer these into a search box, but you'll get a fuzzy, ranked approximation of a question
that has a *precise, structural* answer. These are **graph** questions — about entities and the
relationships between them — and that is the fourth and final layer of Semantic Core Service.

---

## Ranked lists vs. graphs: a different shape of question

A vector index answers **similarity**: "what is *like* this?" A graph answers **connection**:
"what is *related* to this, and by what relationship?" The difference is not cosmetic.

| | Ranked retrieval | Semantic graph |
| --- | --- | --- |
| Question | "what's *similar*?" | "what's *connected*?" |
| Result | ordered `{id, score}` | nodes + typed edges |
| Strength | fuzzy, semantic recall | precise, structural traversal |
| Example | "uplifting space movie" | "all titles by this director in this franchise" |

The mature move isn't to pick one. It's to run both planes from one app and let each answer the
questions it's actually good at — which is exactly how the platform is wired (`app/main.py`
initializes the retrieval services *and* the graph services into the same `app.state`).

---

## Turning content into a graph, automatically

The friction with graphs is usually construction: somebody has to define nodes and draw edges. The
platform removes most of that friction by **deriving structure from metadata on ingest**. When you
post an object to `/semantic/ingest`, a metadata extractor and a relation linker
(`app/semantic/linker.py`) materialize entity nodes and connect the object to them with typed edges:

- a `people` value → a `person` node, linked by a **`contains`** edge
- a `location` value → a `location` node, linked by **`occurs_at`**
- an `event` value → an `event` node, linked by **`happens_during`**
- a `timestamp` → a `time` node, linked by **`captured_at`**

So this ingest call:

```jsonc
POST /semantic/ingest
{
  "id": "scs-04",
  "type": "document",
  "metadata": {
    "title": "Hybrid retrieval: fusing BM25 and vector search",
    "people": ["MLEngineer"],
    "location": "hybrid_search",
    "event": "E3 Hybrid Retrieval"
  }
}
```

doesn't store one row. It produces a little neighbourhood of graph:

```
                 ┌─ contains ───────►  (person)  MLEngineer
   (document)    │
    scs-04 ──────┼─ occurs_at ──────►  (location) hybrid_search
                 │
                 └─ happens_during ─►  (event)    E3 Hybrid Retrieval
```

Query the node and the structure is right there — verbatim from a live instance:

```jsonc
GET /semantic/node/scs-04?direction=outbound
{
  "node": { "id": "scs-04", "type": "document",
            "metadata": { "people": ["MLEngineer"], "location": "hybrid_search",
                          "event": "E3 Hybrid Retrieval" } },
  "neighbors": [
    { "id": "person_mlengineer",        "type": "person"   },
    { "id": "event_e3_hybrid_retrieval","type": "event"    },
    { "id": "location_hybrid_search",   "type": "location" }
  ]
}
```

On top of the explicit links, a **linking engine** (`app/linking/engine.py`) scores nearby nodes on
each ingest and auto-creates `related_to` edges above a threshold — so the graph grows *denser* on
its own as more content arrives, without anyone drawing edges by hand.

---

## Answering structural questions

Once the graph exists, two query styles answer the questions from the top of this post.

**Anchor queries** (`/semantic/query`) resolve an entity and walk *inbound* edges to find what
references it (`app/semantic/filters.py`). "Every document under the E3 milestone" becomes:

```python
event_id = "event_e3_hybrid_retrieval"
docs = graph.get_predecessors(event_id, "happens_during")   # who happens_during this event?
```

Supply several anchors and the candidate sets are **intersected** — *documents owned by the ML
engineer **and** in the hybrid-search module* — which is set algebra over edges, not score
juggling. When no anchor is given, it falls back to a type scan ("all `document` nodes"), so the
same endpoint serves both "filtered by relationship" and "list everything of this type."

**Traversal queries** (`/graph_query/execute`) start from anchors and do a bounded breadth-first
walk. Ask for everything the ML engineer touched, and on the live golden-example graph you get:

```jsonc
POST /graph_query/execute   { "person": "MLEngineer", "k": 25 }
→ { "total": 8,
    "traversal_steps": [ { "step": "media", "count": 8 } ],
    "truncated": false,
    "nodes": [ "nfx-10", "fit-07", "scs-13", "scs-05",
               "scs-07", "scs-06", "nfx-06", "scs-04" ] }
```

Look at what that result *is*: eight documents pulled from **all three projects** in one hop,
because the `MLEngineer` node is shared across them. A ranked list would never have given you that
clean cross-project set — it isn't a similarity question, it's a connection question, and the graph
answers it exactly.

---

## The hard part: keeping traversal from exploding

Graph traversal has a famous failure mode: on a dense graph, an unbounded BFS fans out until it has
visited everything and your "query" is a denial-of-service against your own database. A graph
retrieval layer is only production-safe if it is **bounded by construction**, and the engine
(`app/graph_query/engine.py`) enforces three hard limits:

```python
MAX_NODES = 200          # stop after visiting this many nodes
MAX_DEPTH = 2            # never traverse deeper than this
traversal_steps = plan.get("traversal", [])[:MAX_DEPTH]   # truncate the plan itself
```

When a bound is hit, the walk stops and the response says so honestly:

```python
if len(visited) >= MAX_NODES:
    truncated = True
    break
```

That `truncated` flag ships back to the caller, and every traversal logs its `traversal_steps`
(node count per hop) as a structured event — the same observability discipline from Part 2, applied
to the graph. You can *see* the shape of a walk, and you know when a result was cut short rather
than silently returning a partial answer as if it were complete.

One more design choice worth calling out: when an anchor doesn't resolve, the engine returns an
**empty set, not an error**, and signals `fallback_to_retrieval`. That lets a caller try the graph
first and gracefully fall back to vector retrieval when the question turns out to be a similarity
question after all. The two planes cooperate instead of competing.

---

## Local-first here too: SQLite now, Neo4j when you need it

Consistent with the rest of the platform, the graph runs on **SQLite by default** — a real graph,
persisted to `data/graph.db`, with no service to stand up. The `GraphDB` interface has a `Neo4jGraph`
implementation behind the same contract, so you move to a dedicated graph database by flipping
`GRAPH_TYPE=neo4j` — and if Neo4j is unreachable, it falls back to SQLite rather than failing. You
get to *prototype* graph retrieval with zero infrastructure and *scale* it without touching query
code.

---

## The platform, indexed by itself

The three golden datasets make all of this tangible. Every document across `semantic_core`,
`netflix_clone`, and `fitness_tracker` is tagged with a role, a module, and a milestone — which
means seeding them builds a real engineering knowledge graph: **37 document nodes wired into the
roles, subsystems, and milestones they belong to, ~90 nodes in total.** With it live you can ask:

```bash
# every document in a given module, across the codebase
curl -X POST :8000/semantic/query -d '{"location":"playback_engine","k":25}'

# everything a role delivered, anywhere in the project
curl -X POST :8000/graph_query/execute -d '{"person":"StreamingEngineer","k":25}'

# the work that shipped under one milestone
curl -X POST :8000/semantic/query -d '{"event":"E3 Hybrid Retrieval","k":25}'
```

The platform indexing the story of its own construction is a fitting place to end the series: it is
both the demo *and* the thing being demoed.

---

## The series in one picture

```
   ┌──────────────────────────  one FastAPI app  ──────────────────────────┐
   │                                                                        │
   │   Retrieval plane                         Semantic graph plane         │
   │   ───────────────                         ────────────────────         │
   │   Part 1  vector retrieval                Part 4  entity extraction    │
   │   Part 2  hybrid (BM25 + vectors)                 + auto-linking       │
   │   Part 3  feedback learning + planner             bounded traversal    │
   │                                                                        │
   │   "what is similar?"                       "what is connected?"        │
   └────────────────────────────────────────────────────────────────────────┘
        every layer feature-flagged · local-first · degrades gracefully
```

Four layers, one app, each optional, each observable, each able to grow from a laptop to real
infrastructure behind a stable interface. That layered honesty — start simple, stay debuggable,
scale additively — is the whole point. A vector-search demo shows you can call a model. A system
like this shows you can build the thing that has to *keep working* after the demo ends.

---

## Takeaways

- **Not every question is a ranking question.** "What's connected?" wants a graph, not a score.
- **Derive the graph from metadata** so construction isn't a manual chore, and let an auto-linker
  densify it over time.
- **Bound every traversal** — max nodes, max depth, an honest `truncated` flag — or BFS will eat
  your graph.
- **Run both planes from one app** and let each answer what it's good at, with graceful fallback
  between them.

Thanks for reading the series. The whole system is open source — seed the golden examples, flip the
flags one at a time, and read the logs. They were written to be read.

---

*Part 4 of the Semantic Core Service series. Code references: `app/semantic/service.py`,
`app/semantic/linker.py`, `app/semantic/filters.py`, `app/linking/engine.py`,
`app/graph_query/engine.py`, `app/graph/sqlite_graph.py`.*

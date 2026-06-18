---
title: "Teaching Search to Learn: Click Feedback, Learning-to-Rank, and an Adaptive Planner"
subtitle: "Your best ranking signal is the one users hand you on every click. Here's how to capture it — without waiting for a trained model."
tags: [Learning to Rank, Machine Learning Systems, Search Ranking, Feedback Loops, Recommendation Systems]
series: "Semantic Core Service"
part: 3
---

# Teaching Search to Learn: Click Feedback, Learning-to-Rank, and an Adaptive Planner

[Part 2](02-hybrid-retrieval-bm25-and-vectors.md) ended with a good *static* ranking: hybrid
retrieval gives every query a sensible ordering. But a static ranker has a ceiling. It will make
the same mistake on the same query forever, even as thousands of users vote — with their clicks —
on what the right answer actually was.

The third layer of Semantic Core Service is about catching those votes and acting on them. And
the design decision I want to defend in this post is a contrarian one: **start with rules, not a
model.**

---

## The cold-start problem that sinks "just train a model"

The textbook answer to "make search learn" is learning-to-rank: log impressions and clicks, build
a labelled dataset, train a gradient-boosted ranker (LambdaMART and friends), deploy it, retrain
nightly. It works — *eventually*. The catch is **cold start**:

- A new model needs thousands of labelled events before it generalizes.
- On day one you have *zero* events.
- A freshly deployed system therefore can't learn from the first click, the first hundred clicks,
  or the first day. It just… waits.

That is a brutal trade for a platform whose whole identity (Part 1) is "useful immediately." So
the system does both, layered: a **rule-based learner that improves from the very first click**,
and an **optional ML model** that takes over once enough data exists. The rule-based layer is the
floor; the model is the ceiling.

---

## Step 1: capture the feedback

A click is recorded through one endpoint, with everything you need to learn from it:

```jsonc
POST /feedback
{
  "query": "tracking workouts without internet",
  "namespace": "fitness_tracker",
  "results": ["doc-a", "doc-b", "doc-c"],   // what was shown
  "clicked": "doc-b",                         // what they chose
  "position": 1                               // where it ranked (0-based)
}
```

Two things happen. The event is appended to a durable log (`data/feedback.jsonl`) so nothing is
lost and models can be recomputed later, and the in-memory rule-based learner updates immediately.
The position matters as much as the click: a click on result #5 is a stronger signal that your
ranking was wrong than a click on result #1.

A subtle but important detail in `app/main.py`: on startup, the rule-based learner **replays the
entire feedback log** to rebuild its state. Learning survives restarts without a database:

```python
historical = feedback_store.load_all()
if historical:
    l2r_rule.load_from_feedback(historical)   # bootstrap from every past click
```

---

## Step 2: the rule-based learner (boost what gets clicked)

The rule-based model (`app/learning/l2r_rule.py`) is deliberately tiny. It keeps a namespaced count
of which documents got clicked for which queries:

```python
# {namespace: {query: {doc_id: click_count}}}
```

and at rank time it adds a bounded boost proportional to a document's share of clicks for that
query:

```python
total = sum(query_clicks.values()) or 1
click_boost = _LAMBDA * (click_count / total)   # _LAMBDA = 0.2
final_score = base_score + click_boost
```

That's the whole algorithm, and every choice in it is intentional:

- **It works from click #1.** No training, no batch job — the next search reflects the last click.
- **The boost is bounded** (`_LAMBDA = 0.2`). Feedback *nudges* the order; it never lets one
  popular result steamroll genuine relevance. This is the guardrail against a feedback loop that
  ossifies into "the rich get richer."
- **It's a normalized share, not a raw count.** A doc clicked 3 of 4 times for a query earns more
  than one clicked 3 of 300, even though both have a count of 3.
- **It's namespace-scoped.** Clicks in `netflix_clone` never warp ranking in `fitness_tracker`.
- **It's observable.** Every applied boost logs an `l2r_applied` event with base score, boost, and
  final score, so you can audit exactly why an order changed.

Then the optional ML model (`app/learning/l2r_model.py`) layers on top *only when it's ready*:

```python
if learning_cfg["rule_based"]:
    hits = l2r_rule.rerank(query, namespace, hits)
if learning_cfg["ml_model"] and l2r_model.is_ready:   # graceful fallback if not trained
    hits = l2r_model.rerank(query, hits)
```

No trained model on disk? `is_ready` is false and the system runs happily on rules. That's the
Part 1 graceful-degradation rule showing up again, exactly where a naive ML pipeline would throw.

---

## Step 3: close the loop with an adaptive planner

Here's the part I find most fun. Learning-to-rank reorders results *after* retrieval. But what if
feedback could change *how you retrieve in the first place*?

Recall from Part 2 that hybrid retrieval has knobs: `alpha` (semantic-vs-lexical weight),
`vector_k`, `bm25_k`. The **rule-based planner** sets them from the query's shape
(`app/planner/rule_based.py`):

```python
if query_type == "keyword":     alpha, vector_k, bm25_k = 0.3, 10, 30   # lean lexical
elif query_type == "semantic":  alpha, vector_k, bm25_k = 0.8, 30, 10   # lean vector
else:                           alpha, vector_k, bm25_k = 0.6, 20, 20   # balanced
```

The **adaptive planner** (`app/planner/adaptive.py`) starts from those rules, then bends them using
that query's *measured history* — its click-through rate and mean reciprocal rank:

```python
if ctr < 0.3:   plan["bm25_k"] = min(plan["bm25_k"] + 10, 50)   # few clicks → widen lexical recall
if mrr < 0.4:   plan["alpha"]  = min(plan["alpha"] + 0.1, 1.0)  # clicks rank low → trust vectors more
```

Read those two lines as hypotheses the system tests automatically:

- **Low CTR** (people barely click anything) → the candidate set is probably too narrow, so pull
  more lexical candidates.
- **Low MRR** (people click, but only on results buried down the page) → the *ordering* is wrong,
  so shift weight toward the semantic signal.

Both adjustments are **clamped** (`bm25_k ≤ 50`, `alpha ≤ 1.0`) so the loop can lean but never run
away. And the loop is genuinely closed: `POST /feedback` computes `ctr` and `mrr` for the query and
writes them back to the planner store (`data/planner_store.json`), so the *next* time that query
arrives, retrieval strategy itself has already adapted.

```
   /similar  ──►  retrieve  ──►  rank  ──►  user clicks
       ▲                                        │
       │                                        ▼
   plan (alpha, k) ◄── planner_store ◄──  /feedback (ctr, mrr)
```

---

## Why "rules first" is the senior call

It would have been more impressive-sounding to lead with a trained ranker. It would also have been
worse engineering. The rule-based layer gives you four things a model can't on day one:

1. **Immediate value** — it learns from the first interaction.
2. **A safe fallback** — when the model isn't trained or fails to load, search still improves.
3. **Interpretability** — every reorder is a logged, explainable boost, not a black box.
4. **A baseline** — you cannot honestly claim your fancy model helped until you've measured it
   against the cheap rule it has to beat.

The ML model isn't the hero of this story; it's the *upgrade path*. The rule-based learner and the
adaptive planner are what make the system useful while that data accumulates. Knowing which one to
ship first is the difference between a demo and a system.

---

## See it move

```bash
LEARNING_ENABLED=true PLANNER_ENABLED=true PLANNER_MODE=adaptive \
  uvicorn app.main:app --reload
python scripts/load_golden_examples.py
```

Search, click, search again, and watch the order shift toward what you chose:

```bash
curl -X POST :8000/similar  -d '{"query":"how are recommendations generated","k":3,"namespace":"netflix_clone"}'
curl -X POST :8000/feedback -d '{"query":"how are recommendations generated","namespace":"netflix_clone","clicked":"<id>","position":2}'
curl -X POST :8000/similar  -d '{"query":"how are recommendations generated","k":3,"namespace":"netflix_clone"}'
# the clicked result has moved up; grep the logs for "l2r_applied" to see the boost
```

This is, incidentally, the exact pattern a streaming service uses on its home page — surface
candidates, watch what gets played, let engagement reshape tomorrow's ordering. The `netflix_clone`
golden dataset spells that loop out, and here it's running for real on 80 lines of dependency-free
Python.

---

## Takeaways

- **Cold start is the real enemy.** Ship a rule-based learner that improves from click #1; treat
  the ML model as the upgrade, not the foundation.
- **Bound your boosts.** Feedback should nudge ranking, not let popularity bulldoze relevance.
- **Persist the log, replay on boot.** Learning that survives a restart needs no database.
- **Close the loop into retrieval.** An adaptive planner lets yesterday's clicks change *how* you
  search today — with clamps so it can lean but never spiral.

In [Part 4](04-beyond-ranked-lists-a-semantic-graph.md) I leave ranked lists behind entirely. Some
questions — "show me everything this engineer touched across the project" — aren't ranking problems
at all. They're graph problems. Here's the semantic graph that answers them.

---

*Part 3 of the Semantic Core Service series. Code references: `app/learning/l2r_rule.py`,
`app/learning/l2r_model.py`, `app/planner/rule_based.py`, `app/planner/adaptive.py`,
`app/api/routes.py`.*

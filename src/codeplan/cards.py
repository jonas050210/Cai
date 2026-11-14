"""Concept cards: dense programming knowledge as text, English and German.

A 100M model cannot read enough raw code to learn "when do I use a hash map".
It can, however, learn a few hundred well-formed fact cards. Each card carries
one fact pair (what it is / when to use it / what it costs), written in both
languages, so the same knowledge also produces translation and correction pairs.

Everything here is generated from the tables below - edit a row, get cards - and
the output is plain JSONL the normal pipeline can filter and pack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# name, German name, typical operation costs, when to use, when not to
STRUCTURES = [
    ("array", "Array", "index O(1), search O(n), insert at front O(n)",
     "fixed count, fast sequential access, cache friendly",
     "frequent inserts/removes in the middle",
     "feste Anzahl, schneller sequenzieller Zugriff, cache-freundlich",
     "häufiges Einfügen/Löschen in der Mitte"),
    ("dynamic array (list, vector)", "dynamisches Array (Liste, Vector)",
     "append amortised O(1), index O(1), insert O(n)",
     "growable collection read mostly in order",
     "many random inserts in the middle",
     "wachsende Sammlung, überwiegend der Reihe nach gelesen",
     "viele Einfügungen an zufälligen Positionen"),
    ("linked list", "verkettete Liste", "index O(n), insert/remove given node O(1)",
     "frequent splice operations, unknown size at compile time",
     "iteration speed or cache locality matters",
     "häufiges Umbauen der Liste, Größe zur Laufzeit offen",
     "Iterationsgeschwindigkeit oder Cache-Lokalität zählt"),
    ("hash map (dict, object)", "Hash-Map (Dict, Objekt)",
     "lookup/insert/delete O(1) average, O(n) worst case",
     "key-value lookup, counting, deduplication",
     "ordered iteration or range queries",
     "Schlüssel-Wert-Zugriff, Zählen, Duplikate entfernen",
     "sortierte Iteration oder Bereichsabfragen"),
    ("balanced tree (red-black, AVL)", "balancierter Baum (Rot-Schwarz, AVL)",
     "lookup/insert/delete O(log n), in-order iteration",
     "sorted data with steady inserts, range queries",
     "pure lookup speed matters more than order",
     "sortierte Daten mit laufenden Einfügungen, Bereichsabfragen",
     "reine Lookup-Geschwindigkeit wichtiger als Ordnung"),
    ("binary heap (priority queue)", "Binärheap (Priority Queue)",
     "push/pop O(log n), peek O(1)",
     "scheduling, Dijkstra, top-k problems",
     "you need arbitrary element access",
     "Scheduling, Dijkstra, Top-k-Aufgaben",
     "beliebiger Zugriff auf Elemente nötig"),
    ("trie (prefix tree)", "Trie (Präfixbaum)", "lookup O(key length)",
     "autocomplete, prefix search, IP routing",
     "memory is tight and keys are short",
     "Autovervollständigung, Präfixsuche, IP-Routing",
     "Speicher knapp und Schlüssel kurz"),
    ("set (hash set)", "Set (Hash-Set)", "add/contains O(1) average",
     "membership tests, deduplication",
     "you also need the associated values",
     "Zugehörigkeit prüfen, Duplikate entfernen",
     "die zugehörigen Werte werden gebraucht"),
    ("queue (FIFO)", "Queue (FIFO)", "enqueue/dequeue O(1)",
     "breadth-first search, job pipelines",
     "you need priority ordering",
     "Breitensuche, Auftrags-Pipelines",
     "Prioritätsreihenfolge nötig"),
    ("stack (LIFO)", "Stack (LIFO)", "push/pop O(1)",
     "undo history, parsing, recursion replacement",
     "elements must leave in insertion order",
     "Undo-Verlauf, Parser, Rekursion ersetzen",
     "Elemente müssen in Einfügereihenfolge raus"),
    ("ring buffer", "Ringpuffer", "push/pop O(1), no allocation",
     "fixed-size streaming windows, audio, metrics",
     "the data can grow without bound",
     "Streaming-Fenster fester Größe, Audio, Metriken",
     "Datenmenge unbegrenzt wächst"),
    ("bloom filter", "Bloom-Filter", "add/contains O(k) with k hash functions",
     "cheap pre-filter before an expensive lookup",
     "you must never have false positives",
     "billiger Vorfilter vor einem teuren Lookup",
     "es darf niemals Fehlpositive geben"),
    ("union-find (disjoint set)", "Union-Find (Disjoint Set)",
     "find/union near O(1) amortised",
     "connected components, Kruskal, clustering",
     "you need the actual edges afterwards",
     "zusammenhängende Komponenten, Kruskal, Clustering",
     "die konkreten Kanten werden später gebraucht"),
    ("graph (adjacency list)", "Graph (Adjazenzliste)", "edge iteration O(deg), space O(V+E)",
     "sparse networks, dependency graphs",
     "the graph is nearly complete (use a matrix)",
     "dünn besetzte Netze, Abhängigkeitsgraphen",
     "der Graph fast vollständig ist (Matrix nutzen)"),
]

ALGORITHMS = [
    ("binary search", "Binäre Suche", "O(log n)", "sorted data, exact or boundary search",
     "data is unsorted or changes constantly",
     "sortierte Daten, exakte Suche oder Grenzsuche",
     "Daten unsortiert sind oder sich ständig ändern"),
    ("merge sort", "Mergesort", "O(n log n) always, needs extra memory",
     "stable sorting, linked lists, external sorting",
     "memory is tight",
     "stabile Sortierung, verkettete Listen, externes Sortieren",
     "Speicher knapp ist"),
    ("quicksort", "Quicksort", "O(n log n) average, O(n²) worst case",
     "in-place sorting of arrays",
     "adversarial input or stability is required",
     "Sortieren von Arrays an Ort und Stelle",
     "feindliche Eingaben oder Stabilität gefordert sind"),
    ("heap sort", "Heapsort", "O(n log n) always, in place",
     "memory-constrained sorting",
     "stability or cache friendliness matters",
     "Sortieren mit knappem Speicher",
     "Stabilität oder Cache-Freundlichkeit zählt"),
    ("counting sort", "Countingsort", "O(n + k)",
     "small integer ranges",
     "keys are strings or huge ranges",
     "kleine ganzzahlige Wertebereiche",
     "Schlüssel Strings oder riesige Bereiche sind"),
    ("breadth-first search", "Breitensuche", "O(V + E)",
     "shortest path in unweighted graphs",
     "edges have different weights",
     "kürzeste Wege in ungewichteten Graphen",
     "Kanten unterschiedliche Gewichte haben"),
    ("depth-first search", "Tiefensuche", "O(V + E)",
     "cycle detection, topological order, connectivity",
     "you need shortest distances",
     "Zyklen finden, topologische Ordnung, Zusammenhang",
     "kürzeste Abstände gebraucht werden"),
    ("dijkstra", "Dijkstra", "O((V + E) log V) with a heap",
     "shortest path with non-negative weights",
     "negative edge weights exist",
     "kürzeste Wege mit nicht-negativen Gewichten",
     "negative Kantengewichte vorkommen"),
    ("dynamic programming", "dynamische Programmierung",
     "states × transitions",
     "overlapping subproblems, optimal substructure",
     "the subproblem structure is unclear",
     "überlappende Teilprobleme, optimale Teilstruktur",
     "die Teilstruktur unklar ist"),
    ("memoisation", "Memoisierung", "one cache lookup per call",
     "pure functions with repeated arguments",
     "results depend on hidden state",
     "reine Funktionen mit wiederkehrenden Argumenten",
     "Ergebnisse von verstecktem Zustand abhängen"),
    ("two pointers", "zwei Zeiger", "O(n)",
     "sorted arrays, palindromes, sliding windows",
     "the order of elements is irrelevant",
     "sortierte Arrays, Palindrome, Sliding Windows",
     "die Reihenfolge der Elemente egal ist"),
    ("sliding window", "Sliding Window", "O(n)",
     "contiguous subarray/substring problems",
     "the window cannot be expressed as a range",
     "zusammenhängende Teilfolgen und Teilstrings",
     "sich das Fenster nicht als Bereich ausdrücken lässt"),
]

CONCEPTS = [
    ("big-O notation", "Big-O-Notation",
     "growth of runtime in the input size, ignoring constants",
     "comparing algorithms independently of hardware"),
    ("cache locality", "Cache-Lokalität",
     "data used together should sit together in memory",
     "why an array often beats a linked list in practice"),
    ("hash collision", "Hash-Kollision",
     "two keys mapping to the same bucket",
     "worst-case O(n) lookups and hash-flooding attacks"),
    ("immutability", "Unveränderlichkeit",
     "values are never modified, only replaced",
     "safe sharing between threads, simpler reasoning"),
    ("idempotence", "Idempotenz",
     "running something twice has the same effect as once",
     "retries, HTTP PUT, queue consumers"),
    ("race condition", "Race Condition",
     "outcome depends on timing of concurrent access",
     "why locks, atomics or message passing exist"),
    ("deadlock", "Deadlock",
     "two parties wait for each other's locks forever",
     "ordering lock acquisition or using timeouts"),
    ("N+1 query problem", "N+1-Abfrageproblem",
     "one query for the list, one per element",
     "preloading or joining instead"),
    ("connection pool", "Connection-Pool",
     "reuse expensive connections instead of opening them per request",
     "database and HTTP client defaults"),
    ("backpressure", "Backpressure",
     "a slow consumer signals the producer to slow down",
     "streams, queues, rate limiting"),
    ("index (database)", "Index (Datenbank)",
     "a sorted structure for fast lookups at the cost of writes",
     "why every index slows down inserts"),
    ("normalisation", "Normalisierung",
     "store each fact once, reference by key",
     "avoiding update anomalies"),
    ("migration", "Migration",
     "a versioned, reversible schema change",
     "deploying database changes safely"),
    ("feature flag", "Feature-Flag",
     "runtime switch for unfinished or risky behaviour",
     "releasing without redeploying"),
    ("blue-green deployment", "Blue-Green-Deployment",
     "two identical environments, traffic switched at once",
     "rollback without downtime"),
]

WEB = [
    ("semantic HTML", "semantisches HTML",
     "use the element that describes the content (nav, main, button)",
     "accessibility and screen readers"),
    ("CSS flexbox", "CSS Flexbox",
     "one-dimensional layout along a main axis",
     "rows or columns that stretch and wrap"),
    ("CSS grid", "CSS Grid",
     "two-dimensional layout with explicit tracks",
     "page layouts and dashboards"),
    ("responsive design", "responsives Design",
     "layout adapts to viewport width via media queries",
     "mobile and desktop from one codebase"),
    ("debounce", "Debounce",
     "run only after input stops for a while",
     "search fields, resize handlers"),
    ("throttle", "Throttle",
     "run at most once per interval",
     "scroll and pointer handlers"),
    ("virtual DOM", "virtuelles DOM",
     "diff a lightweight copy, patch the real DOM",
     "why frameworks avoid full re-renders"),
    ("server-side rendering", "Server-Side-Rendering",
     "HTML is produced on the server, hydrated on the client",
     "fast first paint and SEO"),
    ("lazy loading", "Lazy Loading",
     "load code or data only when needed",
     "startup time of large frontends"),
    ("CORS", "CORS",
     "the server must explicitly allow cross-origin requests",
     "why fetch fails between different origins"),
]

PERFORMANCE = [
    ("array iteration vs linked list", "Array-Iteration vs. verkettete Liste",
     "arrays are usually 3-10x faster because of cache locality",
     "prefer contiguous memory in hot loops"),
    ("hash lookup vs linear search", "Hash-Lookup vs. lineare Suche",
     "O(1) vs O(n): at 1000 elements about 50x faster",
     "build a map before searching repeatedly"),
    ("JSON vs Protobuf", "JSON vs. Protobuf",
     "Protobuf is typically 3-10x smaller and faster to parse",
     "hot internal APIs versus debuggable public ones"),
    ("string concatenation vs builder", "String-Konkatenation vs. Builder",
     "repeated concatenation copies the whole string each time",
     "join or use a buffer in loops"),
    ("N+1 queries vs one join", "N+1-Abfragen vs. ein Join",
     "1000 queries can be 100x slower than one",
     "preload relations"),
    ("sync vs async I/O", "synchrones vs. asynchrones I/O",
     "async keeps threads free while waiting",
     "many concurrent network calls"),
]

DOMAIN = "planning"


def _card(kind: str, en_name: str, de_name: str, fact_en: str, fact_de: str,
          detail_en: str, detail_de: str) -> tuple[str, str]:
    english = f"{en_name}: {fact_en}. {detail_en}."
    german = f"{de_name}: {fact_de}. {detail_de}."
    return english, german


def _entries():
    """Yield (kind, english name, german name or None, english text, german text or None)."""
    for name, de, costs, use, avoid, use_de, avoid_de in STRUCTURES:
        yield ("structure", name, de,
               f"{name}: operations cost {costs}. Use it when {use}; avoid it when {avoid}.",
               f"{de}: Operationen kosten {costs}. Nutze es, wenn {use_de}; vermeide es, "
               f"wenn {avoid_de}.")
    for name, de, costs, use, avoid, use_de, avoid_de in ALGORITHMS:
        yield ("algorithm", name, de,
               f"{name}: complexity {costs}. Use it for {use}; not for {avoid}.",
               f"{de}: Komplexität {costs}. Nutze es für {use_de}; nicht, wenn {avoid_de}.")
    # Concepts, web and performance stay English-only: a half-English "German"
    # card would teach the model to mix languages.
    for name, _de, fact, why in CONCEPTS:
        yield ("concept", name, None, f"{name}: {fact}. It matters for {why}.", None)
    for name, _de, fact, why in WEB:
        yield ("web", name, None, f"{name}: {fact}. It matters for {why}.", None)
    for name, _de, fact, why in PERFORMANCE:
        yield ("performance", name, None, f"{name}: {fact}. Rule of thumb: {why}.", None)


def cards() -> list[dict]:
    """Return one record per card, English plus German where the German is real."""
    records = []
    for kind, en_name, de_name, text_en, text_de in _entries():
        for lang, text, name in (("en", text_en, en_name), ("de", text_de, de_name)):
            if text is None:
                continue
            digest = hashlib.sha256(f"card:{kind}:{name}:{lang}".encode()).hexdigest()[:16]
            records.append(
                {
                    "id": digest,
                    "text": text,
                    "lang": lang,
                    "domain": DOMAIN,
                    "license": "cc0-1.0",
                    "source": f"card:{kind}/{name.replace(' ', '_')}.{lang}",
                    "kind": kind,
                }
            )
    return records


def sft_pairs() -> list[dict]:
    """Instruction pairs built from the cards: explain, compare, translate."""
    pairs = []
    for record in cards():
        if record["lang"] != "en":
            continue
        name = record["source"].split("/")[-1].rsplit(".", 1)[0].replace("_", " ")
        pairs.append(
            {
                "system": "You are a precise programming assistant. Answer in one or two sentences.",
                "user": f"What is {name} and when do I use it?",
                "assistant": record["text"],
                "lang": "en",
                "license": "cc0-1.0",
                "source": record["source"],
            }
        )
    for en, de in zip(cards()[0::2], cards()[1::2]):
        if en["lang"] != "en" or de["lang"] != "de":
            continue
        pairs.append(
            {
                "system": "You translate German technical text into English, precisely and without commentary.",
                "user": de["text"],
                "assistant": en["text"],
                "lang": "de-en",
                "license": "cc0-1.0",
                "source": f"translate:{de['source']}",
            }
        )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate concept cards and SFT pairs.")
    parser.add_argument("--out", default="data/raw/cards.jsonl")
    parser.add_argument("--sft-out", default="data/raw/cards_sft.jsonl")
    parser.add_argument("--no-sft", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = cards()
    with open(out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(records)} cards -> {out}")

    if not args.no_sft:
        pairs = sft_pairs()
        sft_out = Path(args.sft_out)
        sft_out.parent.mkdir(parents=True, exist_ok=True)
        with open(sft_out, "w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"{len(pairs)} SFT pairs -> {sft_out}")


if __name__ == "__main__":
    main()

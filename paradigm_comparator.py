"""AI Paradigm Comparator.

Answers the same query with three AI approaches and compares them:

1. Traditional AI - exact rule matching from a dictionary.
2. Generative AI - random combination of knowledge fragments.
3. Agentic AI    - iterative search, self-check and query broadening.

Topic: AI & Machine Learning fundamentals.
Only the Python standard library is used.
"""

import random
import re
from dataclasses import dataclass, field

# =====================================================================
# 1. CONSTANTS AND KNOWLEDGE SOURCES
# =====================================================================

NO_MATCH = "No match found"
MIN_MATCHES = 2      # matches the agent needs to call an answer sufficient
MAX_ITERATIONS = 3   # hard limit on agent search rounds
TOP_K = 3            # entries used to synthesize the final answer

# Traditional AI: normalized question -> fixed answer.
RULES = {
    "what is machine learning": (
        "Machine learning is a branch of AI where systems learn "
        "patterns from data instead of being explicitly programmed."
    ),
    "what is overfitting": (
        "Overfitting is when a model memorizes training data, noise "
        "included, and then performs poorly on new unseen data."
    ),
    "what is a neural network": (
        "A neural network is a model made of layers of connected "
        "neurons that learn weights to map inputs to outputs."
    ),
    "what is supervised learning": (
        "Supervised learning trains a model on labeled examples so it "
        "can predict labels for new inputs."
    ),
    "what is unsupervised learning": (
        "Unsupervised learning finds hidden structure, such as "
        "clusters, in data that has no labels."
    ),
    "what is reinforcement learning": (
        "Reinforcement learning trains an agent to act in an "
        "environment by maximizing cumulative reward."
    ),
    "what is artificial intelligence": (
        "Artificial intelligence is the field of building machines "
        "that perform tasks normally requiring human intelligence."
    ),
    "what is natural language processing": (
        "Natural language processing lets computers understand, "
        "interpret and generate human language."
    ),
    "what is computer vision": (
        "Computer vision enables machines to interpret and understand "
        "images and video."
    ),
}

# Knowledge base shared by the Generative and Agentic approaches.
KNOWLEDGE_BASE = [
    {"id": "ml-1", "tags": {"machine", "learning", "ml"},
     "text": "Machine learning lets computers improve at a task by "
             "learning patterns from data."},
    {"id": "ml-2", "tags": {"machine", "algorithms", "training"},
     "text": "Machine learning algorithms are trained on examples and "
             "then tested on data they have never seen."},
    {"id": "ml-3", "tags": {"machine", "applications", "prediction"},
     "text": "Machine learning powers spam filters, recommendations "
             "and fraud detection."},
    {"id": "sup-1", "tags": {"supervised", "labeled", "classification"},
     "text": "Supervised learning uses labeled data to learn a mapping "
             "from inputs to known outputs."},
    {"id": "uns-1", "tags": {"unsupervised", "clustering", "unlabeled"},
     "text": "Unsupervised learning discovers clusters and structure "
             "in data without labels."},
    {"id": "rl-1", "tags": {"reinforcement", "reward", "agent"},
     "text": "Reinforcement learning teaches an agent by trial and "
             "error, using rewards and penalties."},
    {"id": "nn-1", "tags": {"neural", "network", "networks", "layers"},
     "text": "A neural network stacks layers of artificial neurons that "
             "transform inputs step by step."},
    {"id": "nn-2",
     "tags": {"neural", "networks", "backpropagation", "weights"},
     "text": "Neural networks learn by backpropagation, adjusting "
             "weights to reduce prediction error."},
    {"id": "dl-1", "tags": {"deep", "dl"},
     "text": "Deep learning uses neural networks with many hidden "
             "layers to learn complex features automatically."},
    {"id": "cnn-1", "tags": {"cnn", "convolutional", "image"},
     "text": "CNNs use convolution filters to detect edges and shapes, "
             "which makes them ideal for images."},
    {"id": "over-1",
     "tags": {"overfitting", "generalization", "memorize"},
     "text": "Overfitting happens when a model memorizes the training "
             "set and fails to generalize to new data."},
    {"id": "reg-1", "tags": {"regularization", "dropout", "penalty"},
     "text": "Regularization methods such as dropout and weight decay "
             "reduce overfitting."},
    {"id": "gd-1", "tags": {"gradient", "descent", "optimization"},
     "text": "Gradient descent minimizes a loss function by repeatedly "
             "stepping against the gradient."},
    {"id": "nlp-1", "tags": {"nlp", "language", "text"},
     "text": "NLP lets machines process human language for translation, "
             "chatbots and sentiment analysis."},
    {"id": "trf-1", "tags": {"transformer", "transformers", "attention"},
     "text": "Transformers use self-attention to weigh the relevance "
             "of every word in a sequence."},
    {"id": "cv-1", "tags": {"vision", "computer", "recognition"},
     "text": "Computer vision teaches machines to recognize objects "
             "and scenes in images and video."},
]

STOPWORDS = {
    "what", "is", "a", "an", "the", "how", "do", "does", "explain",
    "from", "of", "to", "in", "on", "are", "about", "me", "tell",
    "why", "can", "and", "or", "with", "for", "give", "describe",
}

# Words too common to be useful in the agent's first, specific search.
GENERIC_WORDS = {"learning", "model", "models"}

# Query expansion used when the agent broadens its search.
SYNONYMS = {
    "deep": ["neural", "layers", "networks"],
    "overfitting": ["generalization", "regularization", "dropout"],
    "neural": ["network", "networks", "layers"],
    "networks": ["neural", "layers"],
    "learn": ["training", "weights", "backpropagation"],
    "data": ["training", "labeled"],
    "language": ["nlp", "text"],
    "vision": ["image", "cnn"],
    "image": ["vision", "cnn"],
    "gradient": ["optimization", "descent"],
}

# 2 queries Traditional AI answers (exact rules), 2 it cannot.
TEST_QUERIES = [
    "What is machine learning?",                 # rule HIT
    "What is overfitting?",                      # rule HIT
    "Explain deep learning",                     # rule MISS
    "How do neural networks learn from data?",   # rule MISS
]


# =====================================================================
# 2. RESULT CONTAINERS
# =====================================================================


@dataclass
class GenerativeResult:
    """Answer from the Generative AI plus an on-topic flag."""

    answer: str
    on_topic: bool


@dataclass
class AgentResult:
    """Answer from the Agentic AI with its reasoning trace."""

    answer: str
    iterations: int
    status: str                      # "OK", "PARTIAL" or "FAIL"
    log: list = field(default_factory=list)


# =====================================================================
# 3. TEXT HELPERS
# =====================================================================


def normalize(text):
    """Lower-case text and strip punctuation and extra spaces."""
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return " ".join(text.split())


def extract_keywords(query):
    """Return query words with stopwords removed (order kept)."""
    words = normalize(query).split()
    return [w for w in words if w not in STOPWORDS]


def specific_keywords(query):
    """Return keywords without generic words (all if none remain)."""
    words = extract_keywords(query)
    return [w for w in words if w not in GENERIC_WORDS] or words


def _unique(words):
    """Remove duplicates from a list while keeping the order."""
    return list(dict.fromkeys(words))


# =====================================================================
# 4. TRADITIONAL AI - EXACT RULE MATCHING
# =====================================================================


def traditional_ai(query):
    """Look the normalized query up in RULES, else 'No match found'."""
    return RULES.get(normalize(query), NO_MATCH)


# =====================================================================
# 5. GENERATIVE AI - RANDOM FRAGMENT COMBINATION
# =====================================================================

CONNECTORS = [
    "Moreover,", "Also,", "In addition,", "Furthermore,",
    "Additionally,", "Notably,",
]


def _join_fragment(text):
    """Lower-case the first letter unless it starts an acronym."""
    first_word = text.split()[0]
    if len(first_word) > 1 and first_word[:2].isupper():
        return text
    return text[0].lower() + text[1:]


def generative_ai(query):
    """Randomly pick and join 2-3 fragments, so output varies."""
    keywords = set(specific_keywords(query))
    relevant = [e for e in KNOWLEDGE_BASE if e["tags"] & keywords]
    others = [e for e in KNOWLEDGE_BASE if e not in relevant]
    pool = relevant + random.sample(others, 2)
    chosen = random.sample(pool, random.choice([2, 3]))
    parts = [chosen[0]["text"]]
    for entry in chosen[1:]:
        connector = random.choice(CONNECTORS)
        parts.append(f"{connector} {_join_fragment(entry['text'])}")
    on_topic = any(e in relevant for e in chosen)
    return GenerativeResult(" ".join(parts), on_topic)


# =====================================================================
# 6. AGENTIC AI - ITERATIVE SEARCH WITH SELF-CHECK
# =====================================================================


def _score(entry, keywords, fuzzy):
    """Count keywords matching an entry (exact tags or substrings)."""
    score = 0
    for word in keywords:
        if word in entry["tags"]:
            score += 1
        elif fuzzy and (
            word in entry["text"].lower()
            or any(len(t) > 2 and (word in t or t in word)
                   for t in entry["tags"])
        ):
            score += 1
    return score


def _search(kb, keywords, fuzzy, core):
    """Return entries with score > 0, best match first.

    Entries matching the original specific keywords (``core``) rank
    above entries that only match broadened ones.
    """
    scored = []
    for entry in kb:
        score = _score(entry, keywords, fuzzy)
        if score > 0:
            rank = (_score(entry, core, fuzzy), score)
            scored.append((rank, entry))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [e for _, e in scored]


def _broaden(keywords, base_keywords):
    """Add synonyms and re-add the generic words that were dropped."""
    wider = list(keywords) + [w for w in base_keywords
                              if w in GENERIC_WORDS]
    for word in base_keywords:
        wider.extend(SYNONYMS.get(word, []))
    return _unique(wider)


def agentic_ai(query, kb=KNOWLEDGE_BASE, verbose=False):
    """Search the KB, self-check, broaden if needed, then synthesize."""
    base = extract_keywords(query)
    core = specific_keywords(query)
    keywords = list(core)
    log = []
    matches = []
    iteration = 0

    def say(message):
        """Record a trace line and print it when verbose."""
        log.append(message)
        if verbose:
            print(message)

    while iteration < MAX_ITERATIONS:
        iteration += 1
        fuzzy = iteration == MAX_ITERATIONS
        mode = "fuzzy" if fuzzy else "exact"
        say(f"Iteration {iteration}: searching {keywords} ({mode})")
        matches = _search(kb, keywords, fuzzy, core)
        count = len(matches)
        noun = "match" if count == 1 else "matches"
        if count >= MIN_MATCHES:
            say(f"  -> Found {count} {noun}. Sufficient!")
            break
        if iteration < MAX_ITERATIONS:
            say(f"  -> Found {count} {noun}. Not enough, broadening...")
            keywords = _broaden(keywords, base)
        else:
            say(f"  -> Found {count} {noun}. Out of iterations.")

    if len(matches) >= MIN_MATCHES:
        status = "OK"
    elif matches:
        status = "PARTIAL"
    else:
        status = "FAIL"
    top = matches[:TOP_K]
    answer = " ".join(e["text"] for e in top) if top else NO_MATCH
    return AgentResult(answer, iteration, status, log)


# =====================================================================
# 7. TEST HARNESS
# =====================================================================


def run_query(query, verbose=True):
    """Run one query through all three approaches.

    Prints each answer (and the agent trace) when ``verbose`` is True
    and returns a comparison row as a dict.
    """
    trad = traditional_ai(query)
    gen = generative_ai(query)
    if verbose:
        print("=" * 70)
        print(f"Query: '{query}'")
        print(f"[Traditional AI] {trad}")
        print(f"[Generative AI] {gen.answer}")
        print("[Agentic AI]")
    agent = agentic_ai(query, verbose=verbose)
    if verbose:
        print(f"Synthesized: {agent.answer}")
        print()
    return {
        "query": query,
        "traditional": "HIT" if trad != NO_MATCH else "MISS",
        "generative": "OK" if gen.on_topic else "WEAK",
        "agentic": f"{agent.status} ({agent.iterations} iter)",
    }


def show_generative_variation(query="What is machine learning?",
                              runs=3):
    """Run Generative AI several times to show output variation."""
    print(f"Generative AI variation for '{query}':")
    for number in range(1, runs + 1):
        print(f"  Run {number}: {generative_ai(query).answer}")
    print()


def print_comparison(rows=None):
    """Print a fixed-width table of which approach succeeded.

    ``rows`` come from run_query(); if omitted the test queries are
    run silently first.
    """
    if rows is None:
        rows = [run_query(q, verbose=False) for q in TEST_QUERIES]
    header = f"{'Query':<42}| {'Traditional':<12}| " \
             f"{'Generative':<11}| Agentic"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['query']:<42}| {row['traditional']:<12}| "
              f"{row['generative']:<11}| {row['agentic']}")
    print()


# =====================================================================
# 8. MAIN PROGRAM
# =====================================================================


def main():
    """Run every demo section of the assignment."""
    print("AI PARADIGM COMPARATOR")
    print(f"Rules: {len(RULES)} | Knowledge base: "
          f"{len(KNOWLEDGE_BASE)} entries\n")
    rows = [run_query(query) for query in TEST_QUERIES]
    show_generative_variation()
    print("COMPARISON TABLE")
    print_comparison(rows)


if __name__ == "__main__":
    main()

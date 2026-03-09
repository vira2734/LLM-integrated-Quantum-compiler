import json, random, re
from tqdm import tqdm

random.seed(7)

# -----------------------
# Graph templates on 0..k-1
# -----------------------
def canon(edges):
    out, seen = [], set()
    for a, b in edges:
        if a == b:
            continue
        u, v = (a, b) if a < b else (b, a)
        if (u, v) not in seen:
            seen.add((u, v))
            out.append([u, v])
    out.sort()
    return out

def path_edges(k):
    return canon([(i, i + 1) for i in range(k - 1)])

def cycle_edges(k):
    return canon([(i, i + 1) for i in range(k - 1)] + [(k - 1, 0)])

def clique_edges(k):
    return canon([(i, j) for i in range(k) for j in range(i + 1, k)])

def star_edges(k):
    return canon([(0, i) for i in range(1, k)])

# -----------------------
# Shape lexicon: words -> (normalized shape label, edge builder)
# IMPORTANT: polygon words map to cycle(k), NOT clique(k)
# -----------------------
def polygon_label(k):
    names = {3:"triangle",4:"square",5:"pentagon",6:"hexagon",7:"heptagon",8:"octagon",9:"nonagon",10:"decagon"}
    return names.get(k, "polygon")

SHAPE_SYNONYMS = [
    # Polygons / cycles by name (shape inferred from words)
    ("triangle",    "triangle",  cycle_edges),
    ("3-cycle",     "triangle",  cycle_edges),
    ("cycle of 3",  "triangle",  cycle_edges),

    ("square",      "square",    cycle_edges),
    ("4-cycle",     "square",    cycle_edges),
    ("cycle of 4",  "square",    cycle_edges),
    ("loop of 4",   "square",    cycle_edges),

    ("pentagon",    "pentagon",  cycle_edges),
    ("5-cycle",     "pentagon",  cycle_edges),
    ("cycle of 5",  "pentagon",  cycle_edges),

    ("hexagon",     "hexagon",   cycle_edges),
    ("6-cycle",     "hexagon",   cycle_edges),

    ("heptagon",    "heptagon",  cycle_edges),
    ("7-cycle",     "heptagon",  cycle_edges),

    ("octagon",     "octagon",   cycle_edges),
    ("8-cycle",     "octagon",   cycle_edges),

    ("nonagon",     "nonagon",   cycle_edges),
    ("9-cycle",     "nonagon",   cycle_edges),

    ("decagon",     "decagon",   cycle_edges),
    ("10-cycle",    "decagon",   cycle_edges),

    # Generic cycle words
    ("ring",        "ring",      cycle_edges),
    ("cycle",       "ring",      cycle_edges),
    ("loop",        "ring",      cycle_edges),
    ("closed loop", "ring",      cycle_edges),

    # Line/path
    ("line",        "line",      path_edges),
    ("chain",       "line",      path_edges),
    ("path",        "line",      path_edges),
    ("linear",      "line",      path_edges),

    # Clique
    ("clique",          "clique", clique_edges),
    ("fully connected", "clique", clique_edges),
    ("all-to-all",      "clique", clique_edges),
    ("complete graph",  "clique", clique_edges),

    # Star
    ("star",        "star",      star_edges),
    ("hub-and-spoke","star",     star_edges),
    ("hub and spokes","star",    star_edges),
]

# For fast sampling by "type"
SYN_BY_KIND = {}
for phrase, norm, builder in SHAPE_SYNONYMS:
    SYN_BY_KIND.setdefault(norm, []).append((phrase, builder))

# -----------------------
# Number forms for k
# -----------------------
NUM_WORD = {3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",10:"ten"}
def k_forms(k):
    return [
        str(k),
        NUM_WORD[k],
        f"{k}-qubit",
        f"{k} qubit",
        f"{k}q",
        f"{NUM_WORD[k]}-qubit",
    ]

# -----------------------
# Text variation utilities
# -----------------------
JOINERS = [", ", "; ", " and ", " / ", " | ", " — ", " ... "]
PREFIXES = [
    "I want", "I need", "Please set", "Configure", "For my compiler,", "Constraint:", "Requirement:"
]
SUFFIXES = [
    "", " Thanks.", " Output JSON only.", " Keep it consistent.", " This is important.", " ASAP."
]
FILLER = [
    "", "", "",  # many empties so filler is occasional
    "Also, ignore scheduling details.",
    "This is for a quantum circuit pass.",
    "I might change this later.",
    "Don't overthink it.",
]

RULE_TEMPLATES = [
    "{kref} gates should be a {shape}.",
    "use a {shape} for {kref} gates",
    "{kref} gate topology = {shape}",
    "for {kref}, connect as {shape}",
    "make {kref} gates follow {shape}",
    "{shape} for {kref} gates",
]

MULTI_TEMPLATES = [
    "{prefix} {rules}{suffix}",
    "{prefix}: {rules}{suffix}",
    "{rules}{suffix}",
    "{prefix} the following. {rules}{suffix}",
    "{prefix} these rules: {rules}{suffix}",
]

# Sometimes add noise like "my" "all my" "every"
def maybe_gate_owner():
    return random.choice(["", "my ", "all my ", "every "])

def make_one_rule(k, shape_phrase):
    kref = random.choice(k_forms(k))
    # sprinkle "my" / "every"
    kref = maybe_gate_owner() + kref
    tmpl = random.choice(RULE_TEMPLATES)
    return tmpl.format(kref=kref, shape=shape_phrase)

def choose_shape_for_k(k):
    # Prefer polygon-named shapes that match k sometimes, but not always.
    # If phrase is "square" with k=5, that's inconsistent; we avoid that.
    polygon_names = {3:"triangle",4:"square",5:"pentagon",6:"hexagon",7:"heptagon",8:"octagon",9:"nonagon",10:"decagon"}
    # 60%: use the matching polygon word for that k
    if random.random() < 0.60:
        word = polygon_names[k]
        # pick a synonym that maps to that normalized polygon label
        options = SYN_BY_KIND[word]
        phrase, builder = random.choice(options)
        return phrase, word, builder
    # otherwise: pick a generic shape kind
    kind = random.choice(["ring","line","clique","star"])
    options = SYN_BY_KIND[kind]
    phrase, builder = random.choice(options)
    return phrase, kind, builder

def normalize_rules(rule_specs):
    """
    rule_specs: list of (k, norm_shape, builder)
    If k repeats, last mention wins (models real prompts).
    Returns sorted list of rule dicts by nQubits.
    """
    latest = {}
    for k, norm_shape, builder in rule_specs:
        edges = builder(k)
        # If polygon label was used, keep that polygon label; otherwise keep norm_shape
        # norm_shape might be "ring", "line", etc.
        shape_label = norm_shape
        # If it's one of the polygon words, ensure label matches k
        if norm_shape in ("triangle","square","pentagon","hexagon","heptagon","octagon","nonagon","decagon"):
            shape_label = polygon_label(k)
        latest[k] = {"nQubits": k, "shape": shape_label, "edges": edges}

    rules = list(latest.values())
    rules.sort(key=lambda r: r["nQubits"])
    return rules

def make_prompt(max_rules=7):
    # number of instructions (any number): sample between 1 and max_rules
    n_rules = random.randint(1, max_rules)

    # choose k values with possible repeats to simulate "override" language
    ks = [random.randint(3, 10) for _ in range(n_rules)]
    rule_texts = []
    rule_specs = []

    for k in ks:
        shape_phrase, norm_shape, builder = choose_shape_for_k(k)
        rule_texts.append(make_one_rule(k, shape_phrase))
        rule_specs.append((k, norm_shape, builder))

    # random ordering, random joiner
    joiner = random.choice(JOINERS)
    rules_blob = joiner.join(rule_texts)

    # optional filler sentences inserted randomly
    filler = random.choice(FILLER)
    prefix = random.choice(PREFIXES)
    suffix = random.choice(SUFFIXES)
    template = random.choice(MULTI_TEMPLATES)

    text = template.format(prefix=prefix, rules=rules_blob, suffix=suffix)
    if filler:
        # insert filler before or after
        if random.random() < 0.5:
            text = filler + " " + text
        else:
            text = text + " " + filler

    # occasional casing / punctuation noise
    if random.random() < 0.15:
        text = text.replace("gates", "GATES")
    if random.random() < 0.10:
        text = text.replace(".", " .")
    if random.random() < 0.10:
        text = text.replace("qubit", "qbit").replace("qubits", "qbits")

    rules = normalize_rules(rule_specs)
    out = json.dumps({"rules": rules}, separators=(",", ":"), ensure_ascii=False)
    return text.strip(), out

def validate_output(out_json):
    obj = json.loads(out_json)
    if set(obj.keys()) != {"rules"}:
        return False
    rules = obj["rules"]
    if not isinstance(rules, list):
        return False
    prev_k = -1
    for r in rules:
        if set(r.keys()) != {"nQubits","shape","edges"}:
            return False
        k = r["nQubits"]
        if not isinstance(k, int) or k < 3 or k > 10:
            return False
        if k <= prev_k:
            return False
        prev_k = k
        edges = r["edges"]
        if edges != sorted(edges):
            return False
        for e in edges:
            if not (isinstance(e, list) and len(e) == 2):
                return False
            u, v = e
            if not (isinstance(u, int) and isinstance(v, int)):
                return False
            if u >= v:
                return False
            if u < 0 or v >= k:
                return False
    return True

def wrap_autotrain(inp, out):
    return {
        "text":
        "### Instruction:\n"
        "Extract ALL gate-size connectivity rules from the input. "
        "Output ONLY valid JSON matching this schema: "
        "{\"rules\":[{\"nQubits\":int,\"shape\":string,\"edges\":[[int,int],...]},...]}. "
        "Use canonical node ids 0..k-1 for each rule.\n"
        "### Input:\n" + inp +
        "\n### Output:\n" + out
    }

def main():
    TARGET = 30000
    MAX_RULES_PER_PROMPT = 10  # "any number" up to 10 in training; generalizes beyond
    seen = set()
    rows = []

    for _ in tqdm(range(TARGET * 2), desc="Generating"):  # oversample to dedupe
        inp, out = make_prompt(max_rules=MAX_RULES_PER_PROMPT)
        if not validate_output(out):
            continue
        key = (inp, out)
        if key in seen:
            continue
        seen.add(key)
        rows.append((inp, out))
        if len(rows) >= TARGET:
            break

    random.shuffle(rows)

    n = len(rows)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    splits = {
        "/Users/vinay/Desktop/desktop-Items/independent-study-2/LLM-based-compiler/LLM-SAT-compiler/dataset-gen/train.jsonl": rows[:n_train],
        "/Users/vinay/Desktop/desktop-Items/independent-study-2/LLM-based-compiler/LLM-SAT-compiler/dataset-gen/valid.jsonl": rows[n_train:n_train+n_val],
        "/Users/vinay/Desktop/desktop-Items/independent-study-2/LLM-based-compiler/LLM-SAT-compiler/dataset-gen/test.jsonl": rows[n_train+n_val:],
    }

    for path, data in splits.items():
        with open(path, "w", encoding="utf-8") as f:
            for inp, out in data:
                f.write(json.dumps(wrap_autotrain(inp, out), ensure_ascii=False) + "\n")

    print("Wrote:", {k: len(v) for k, v in splits.items()}, "total", n)

if __name__ == "__main__":
    main()
from collections import defaultdict

# Example Facts: Each tuple means parent(X, Y) is True
parent_facts = [
    ('bob', 'ann'),
    ('bob', 'pat'),
    ('pat', 'jim'),
    ('tom', 'liz')  # Add more as needed
]

# Generate parent lookup for efficiency
parent_dict = defaultdict(list)
for p, c in parent_facts:
    parent_dict[p].append(c)

# Targets for ancestor
positive_examples = [
    ('bob', 'ann'),
    ('bob', 'pat'),
    ('bob', 'jim'),   # bob->pat->jim
    ('pat', 'jim'),
    ('tom', 'liz')
]
negative_examples = [
    ('ann', 'bob'),  # Not ancestor in this direction
    ('pat', 'bob'),
    ('jim', 'bob'),
    ('liz', 'tom')
]

def covers(rule, example):
    "Check if the rule covers (X, Y)"
    return rule(example[0], example[1])

# FOIL-style Recursive Rule Learner
def ancestor(x, y, visited=None):
    if visited is None:
        visited = set()
    if x in visited:
        return False
    if y in parent_dict[x]:
        return True
    visited.add(x)
    for child in parent_dict[x]:
        if ancestor(child, y, visited):
            return True
    return False

# Evaluate rules
def evaluate_rule(rule, positives, negatives):
    tp = sum([covers(rule, ex) for ex in positives])
    tn = sum([not covers(rule, ex) for ex in negatives])
    return tp, tn

print("Learning ancestor relation using FOIL-style rule induction:\n")
print("Rule: ancestor(X, Y) ← parent(X, Y) OR (parent(X, Z) ∧ ancestor(Z, Y))\n")

tp, tn = evaluate_rule(lambda x, y: ancestor(x, y), positive_examples, negative_examples)
print(f"Positive examples covered: {tp}/{len(positive_examples)}")
print(f"Negative examples NOT covered: {tn}/{len(negative_examples)}")

# Demonstration
test_cases = [('bob', 'jim'), ('tom', 'liz'), ('bob', 'pat'), ('pat', 'bob')]
print("\nTest cases (is ancestor):")
for x, y in test_cases:
    print(f"{x} -> {y}: {ancestor(x,y)}")

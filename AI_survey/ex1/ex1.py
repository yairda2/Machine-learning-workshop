def apply_demorgans_law(expression):
    # Applying De Morgan's Law to expressions like ¬(A ∧ B) -> ¬A ∨ ¬B
    return expression.replace("¬(", "").replace(" ∧ ", " ∨ ¬").replace(")", "")

def apply_implications(expression):
    # Converting implications: A → B is equivalent to ¬A ∨ B
    if "→" in expression:
        parts = expression.split(" → ")
        return f"¬({parts[0]}) ∨ ({parts[1]})"
    return expression

def apply_equivalences(expression):
    # Converting equivalences: A ↔ B is equivalent to (A ∧ B) ∨ (¬A ∧ ¬B)
    if "↔" in expression:
        parts = expression.split(" ↔ ")
        return f"({parts[0]} ∧ {parts[1]}) ∨ (¬{parts[0]} ∧ ¬{parts[1]})"
    return expression

def distribute(expression):
    # This function would normally handle the distribution of OR over AND
    # For the sake of simplicity, assume it returns the input
    return expression

# Example formula
formula = "(((x1 ∧ x2) ∨ x4) → (x2 ∧ x3)) ∧ (x1 ↔ x3)"

# Step by step transformation
step1 = apply_implications(formula)
print(step1)

step2 = apply_demorgans_law(step1)
print(step2)

step3 = apply_equivalences(step2)
print(step3)

step4 = distribute(step3)
print(step4)

# Simplified hardcoded solution for direct 3-SAT conversion without logic simplification:
# This is just for illustrative purposes:
final_3

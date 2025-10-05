import re

# Read the file
with open('adaptive_curve_analysis.py', 'r') as f:
    content = f.read()

# Fix all tensor storage to convert to CPU when storing
content = re.sub(
    r'(\w+_scores\.append\([^)]+\))',
    r'# Convert tensor to CPU before storing\n        \1.cpu().numpy() if hasattr(\1, "cpu") else \1',
    content
)

# Fix budget storage
content = re.sub(
    r'budgets\.append\([^)]+\)',
    r'budgets.append(budget.cpu().numpy() if hasattr(budget, "cpu") else budget)',
    content
)

# Fix best_tradeoff creation
content = re.sub(
    r'best_tradeoff = \{[^}]+\}',
    r'''best_tradeoff = {
        "budget": budgets[optimal_tradeoff_idx].cpu().numpy() if hasattr(budgets[optimal_tradeoff_idx], "cpu") else budgets[optimal_tradeoff_idx],
        "utility_score": utility_scores[optimal_tradeoff_idx].cpu().numpy() if hasattr(utility_scores[optimal_tradeoff_idx], "cpu") else utility_scores[optimal_tradeoff_idx],
        "privacy_score": privacy_scores[optimal_tradeoff_idx].cpu().numpy() if hasattr(privacy_scores[optimal_tradeoff_idx], "cpu") else privacy_scores[optimal_tradeoff_idx],
        "tradeoff_score": tradeoff_scores[optimal_tradeoff_idx].cpu().numpy() if hasattr(tradeoff_scores[optimal_tradeoff_idx], "cpu") else tradeoff_scores[optimal_tradeoff_idx]
    }''',
    content
)

# Write the fixed file
with open('adaptive_curve_analysis.py', 'w') as f:
    f.write(content)

print("Fixed all CUDA tensor issues!")

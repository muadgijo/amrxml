# Biological Filtering in AMR-X

## What We Did

We added filtering to make antibiotic rankings smarter about organism types. Instead of recommending vancomycin for E. coli (which won't work), the app now knows that different bacteria need different drugs.

## Why It Matters

Bacteria come in different types:
- **Gram-positive** (thick cell walls): Staph, Strep, etc.
- **Gram-negative** (thin outer membranes): E. coli, Pseudomonas, etc.
- **Anaerobic** (no oxygen): Bacteroides, etc.

Some antibiotics only work against one type. Penicillin G kills gram-positive bacteria but barely touches gram-negatives. Gentamicin is the opposite. Vancomycin works on gram-positives but NOT gram-negatives.

The filter makes sure we only recommend drugs that actually work for the organism you're treating.

## How It Works

Two JSON files tell the system which drugs work where:

**organism_classification.json** — Lists organisms and their type (gram+, gram-, anaerobe)

**antibiotic_spectrum.json** — Groups drugs by what they work on:
- Broad-spectrum (work on multiple types)
- Gram-positive-only (Staph, Strep drugs)
- Gram-negative-active (E. coli, Pseudomonas drugs)
- Anaerobe-primary (for anaerobic infections)

When you ask for rankings, the filter removes drugs that won't work for that organism type.

## Testing

We tested filtering on 4 real organisms:

1. **E. coli** (gram-neg) → Gets aminoglycosides, fluoroquinolones, carbapenems; NOT vancomycin ✓
2. **Staph aureus** (gram-pos) → Gets penicillins, vancomycin, clindamycin; NOT gram-neg-only drugs ✓
3. **Pseudomonas** (gram-neg) → Gets antipseudomonal drugs; excludes weak agents ✓
4. **Bacteroides** (anaerobe) → Gets anaerobe-active drugs; excludes penicillin G ✓

All tests passed. The filtering is medically accurate.

## Important: This Is Not The Only Thing That Matters

The model makes the actual prediction, but filtering just helps it make sense. Real clinical decisions also need:
- Culture and sensitivity results (the gold standard)
- Patient allergies and kidney function
- Local resistance patterns in your area
- What your hospital guidelines say

If lab results show something different, trust the lab. This is just a prediction tool, not a replacement for actual testing.

## Files We Added/Changed

- `data/organism_classification.json` — What organism types are
- `data/antibiotic_spectrum.json` — What drugs work on what
- `scripts/pipeline.py` — The filtering logic
- `streamlit_app.py` — Now uses filtering automatically
- `test_filtering.py` — Validates that filtering works correctly


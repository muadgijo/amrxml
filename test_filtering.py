#!/usr/bin/env python
"""Test biological filtering across organism types."""

from scripts.pipeline import load_spectrum_rules, filter_by_spectrum

# Load rules
rules = load_spectrum_rules()
if not rules:
    print("ERROR: Spectrum rules not loaded")
    exit(1)

# Test organisms
test_cases = {
    "ESCHERICHIA COLI": {
        "type": "gram-negative",
        "should_have": ["GENTAMICIN", "CIPROFLOXACIN", "MEROPENEM", "CEFOTAXIME"],
        "should_NOT_have": ["VANCOMYCIN", "PENICILLIN", "OXACILLIN", "CLINDAMYCIN"]
    },
    "STAPHYLOCOCCUS AUREUS": {
        "type": "gram-positive",
        "should_have": ["VANCOMYCIN", "PENICILLIN", "OXACILLIN", "CLINDAMYCIN"],
        "should_NOT_have": ["GENTAMICIN"]  # aminoglycosides alone don't work for staph
    },
    "PSEUDOMONAS AERUGINOSA": {
        "type": "gram-negative",
        "should_have": ["CIPROFLOXACIN", "PIPERACILLIN/TAZOBACTAM", "CEFTAZIDIME"],
        "should_NOT_have": ["VANCOMYCIN", "PENICILLIN", "OXACILLIN"]
    },
    "BACTEROIDES FRAGILIS": {
        "type": "anaerobe",
        "should_have": ["CLINDAMYCIN", "METRONIDAZOLE", "MEROPENEM"],
        "should_NOT_have": ["PENICILLIN"]  # penicillin alone doesn't cover anaerobes well
    }
}

# Get all antibiotics
all_abx = [
    "GENTAMICIN", "TOBRAMYCIN", "AMIKACIN", "CEFTRIAXONE", "CEFOTAXIME",
    "CEFTAZIDIME", "CEFEPIME", "CIPROFLOXACIN", "LEVOFLOXACIN", "MEROPENEM",
    "IMIPENEM", "PIPERACILLIN/TAZOBACTAM", "CEFOXITIN", "COLISTIN",
    "VANCOMYCIN", "PENICILLIN", "OXACILLIN", "CEPHALEXIN/CEPHALOTHIN",
    "CEFAZOLIN", "CLINDAMYCIN", "LINEZOLID", "DAPTOMYCIN", "CEFTAROLINE",
    "ERYTHROMYCIN", "CLARITHROMYCIN", "DOXYCYCLINE", "METRONIDAZOLE",
    "AMOXICILLIN/CLAVULANIC ACID", "AMPICILLIN/SULBACTAM", "ERTAPENEM"
]

print("=" * 80)
print("BIOLOGICAL FILTERING VALIDATION")
print("=" * 80)

all_pass = True

for organism, expected in test_cases.items():
    print(f"\n{organism} ({expected['type']}):")
    print("-" * 80)
    
    filtered = filter_by_spectrum(organism, all_abx, rules)
    print(f"  Filtered count: {len(filtered)} / {len(all_abx)} total")
    
    # Check should_have
    missing = []
    for abx in expected["should_have"]:
        if abx not in filtered:
            missing.append(abx)
            all_pass = False
    
    if missing:
        print(f"  ❌ MISSING (should have): {missing}")
    else:
        print(f"  ✅ Has all expected drugs")
    
    # Check should_NOT_have
    unwanted = []
    for abx in expected["should_NOT_have"]:
        if abx in filtered:
            unwanted.append(abx)
            all_pass = False
    
    if unwanted:
        print(f"  ❌ UNWANTED (should exclude): {unwanted}")
    else:
        print(f"  ✅ Excludes inappropriate drugs")
    
    print(f"\n  Available: {filtered[:8]}")

print("\n" + "=" * 80)
if all_pass:
    print("✅ ALL TESTS PASSED - Filtering is medically valid!")
else:
    print("❌ TESTS FAILED - Fix spectrum definitions")
print("=" * 80)

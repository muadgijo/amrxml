# AMR-X Scientific Methodology

## The Core Idea

This model doesn't claim to know what percentage of E. coli are resistant in your country. Instead, it provides a **baseline risk ranking** based on global clinical patterns — useful as a reference point when local data isn't available.

---

## 1. What Actually Transfers Across Regions (and What Doesn't)

### What Transfers: Structural Patterns
- Which antibiotic classes work against which bacteria (mechanism-level stuff)
- Relative ranking of drugs for a given organism (drug A is safer than drug B)
- Which resistance mechanisms are common (ESBLs, carbapenemases, etc.)

**Example:** If the model learns that fluoroquinolones are riskier than aminoglycosides for E. coli, that ranking usually holds globally — even if the absolute percentages differ.

### What Doesn't Transfer: Exact Numbers
- Precise resistance percentages in specific countries
- How fast resistance is changing in your region
- How prescribing patterns locally affect resistance

**Example:** US data might show 35% ciprofloxacin resistance in E. coli, but India could be 65%. The ranking (ciprofloxacin is high-risk for E. coli) probably transfers. The exact number definitely doesn't.

---

## 2. When This Model Actually Works

### For US/Europe
- Use predictions directly — they approximate local patterns
- Compare against your hospital's actual antibiogram to validate

### For India, Africa, Southeast Asia (Limited Data)
- **Use it as:** A starting reference, not gospel truth
- **What you get:** Which drugs are relatively safer/riskier for an organism
- **What you validate against:** Your local resistance data
- **Key assumption:** Real resistance is probably at least as high as the model predicts (safety-oriented)

### Why This Matters
Lots of low-income regions don't have:
- National AMR tracking systems
- Published resistance datasets
- Labs doing routine culture testing

So a **baseline reference** — even if imperfect — is better than nothing. And over time, you can check the model against real data.

---

## 3. Why We Don't Do Time-Series Forecasting

### The Problem
The dataset has no timestamps. We know resistance rates but not *when* those tests happened. Could be 2010, could be 2024.

### Why That Matters
Predicting "resistance will increase X% next year" requires knowing how it changed in the past. Without time labels, that's just made-up.

### What We Do Instead
The model gives you a **snapshot** of current patterns. The app can track:
- How many people are querying carbapenem resistance in Klebsiella (growing concern?)
- Regional query patterns (which areas are interested in which bugs?)
- Trend of user behavior (signals, not predictions)

That's **behavioral epidemiology** — useful for seeing what people are worried about — but it's not the model making temporal forecasts.

---

## 4. How Good Is This Model, Really?

### Numbers
- 68% accuracy (beats random guessing at 63%)
- 0.81 ROC-AUC (decent discrimination)
- 82% sensitivity (catches resistant cases 4 out of 5 times)
- 65% specificity (correctly identifies susceptible cases ~2 out of 3 times)

### What That Means
The model is conservative. It errs toward warning "this might be resistant" rather than falsely reassuring you. For a surveillance/awareness tool, that's the right bias. Missing resistance is worse than a false alarm.

68% isn't "bad" — it reflects that predicting resistance is genuinely hard, especially with minimal information (just organism + antibiotic).

---

## 5. How We Validated It

### Train-Test Split
- 70% of data for training
- 30% for testing (never seen during training)
- Balanced by resistance label (so we test on representative data)

### Why Not Time-Based Split?
We'd need timestamps to say "train on 2010-2015, test on 2016-2020." We don't have that.

### Future Validation
When regional datasets become available:
- Test on India-specific data (does US baseline apply?)
- Train on mixed data from multiple regions
- Add time-indexed data for temporal validation

---

## 6. Why This Scientific Approach Matters

### Common Pitfall
Many published AMR models claim accuracy of 80-90% — but they test on data from the same hospital/region where they trained. Not surprising they do well.

### Our Approach
- ✅ Honest about limitations
- ✅ Clear about transferability
- ✅ Under-promise, over-deliver
- ✅ Encouraging external validation

This is boring but defensible. No one can accuse you of overclaiming.

---

## 7. When Someone Challenges You

### "This doesn't have Indian data!"
Response: "Correct. The model provides structural resistance patterns that often transfer globally. Use it as a baseline reference, validate against your local data, and as you collect local data, we can recalibrate. WHO GLASS does the same thing — global baseline + local adaptation."

### "Why can't you forecast resistance trends?"
Response: "The dataset lacks timestamps, so forecasting would be scientifically dishonest. We give you a current-state snapshot. System-level tracking of query patterns can signal behavioral trends (growing concern about carbapenem resistance) but that's different from model-based time-series forecasting."

### "68% accuracy seems low."
Response: "It exceeds the majority-class baseline and reflects real-world AMR prediction difficulty. The 82% sensitivity means we catch most resistant cases. For a surveillance tool in data-scarce regions, conservative over-prediction is preferable to missing resistance."

---

## 8. Bottom Line

**This model IS scientifically valid because:**
- It doesn't claim more than it delivers
- It matches claims to data capabilities
- It encourages local validation
- It's positioned as a comparative tool, not a diagnostic

**It's NOT because:**
- You're overselling accuracy you don't have
- You're pretending to forecast without time data
- You're treating it as patient-specific guidance

Good science means being honest about what you know and what you don't. This does that.


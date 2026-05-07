## 1. Dataset Overview

Start by grounding the reader:

> This analysis is based on manually evaluated building samples across four countries: **Mexico, Liberia, Niger, and Nepal**.
> Each sample consists of:
>
> * Original building geometry (Google Open Buildings)
> * SAM prediction
> * Postprocessed geometry
>
> The evaluation includes qualitative ratings and categorized geometric errors.

Then explicitly report:

* Total number of evaluated samples
* Number of buildings per country

👉 You’ll compute something like:

```python
df.groupby("country").size()
```

---

## 2. Overall Performance (Baseline vs Postprocessing)

You want a **high-level answer first**:

> Does postprocessing improve building quality?

### Metrics to report:

* Distribution of:

  * `original` ratings
  * `post` ratings
* % of improvements:

  * bad → good/perfect
  * ok → good/perfect
* % of degradations:

  * good → ok/bad

👉 Key statement:

> “Postprocessing improves X% of buildings while degrading Y%.”

---

## 3. Error Categories (Before vs After)

This is one of your strongest sections.

### Compare:

* `original_errors`
* `post_errors`

Focus on:

* Frequency of each error type:

  * SHIFTED
  * SHAPE_MISMATCH
  * MISSING_PARTS
  * EXTRA_PARTS
  * OVERSIMPLIFIED

👉 Example insights:

* SHIFTED likely dominates in original
* EXTRA_PARTS / MISSING_PARTS may increase after postprocessing

### Key question:

> Does postprocessing **remove or introduce** specific error types?

---

## 4. Postprocessing Impact (SAM vs POST)

Use your `post_vs_sam`:

* % where postprocessing introduced **new errors**
* % where it improved geometry

👉 This is critical:

> “Postprocessing introduced new geometric errors in X% of cases.”

---

## 5. Country-Level Differences

Now go deeper — this is where your work becomes interesting.

Compare per country:

### a) Performance

* Average rating (original vs post)
* Improvement rate

### b) Error distribution

* Which errors dominate per country?

Example hypotheses:

* Niger / Liberia:

  * more irregular buildings → more SHAPE_MISMATCH
* Nepal:

  * dense settlements → more MISSING_PARTS
* Mexico:

  * larger buildings → different behavior

---

## 6. Area Analysis

Now bring in geometry:

### Compare:

* Original area vs Post area

Metrics:

* Mean area difference
* % increase / decrease

👉 Key questions:

* Are buildings getting **larger or smaller**?
* Is postprocessing **overgrowing** or **shrinking** footprints?

---

## 7. Shape Complexity

You want to measure if polygons become more detailed.

### Metrics:

* Number of vertices
* Perimeter / area ratio
* Compactness

👉 Questions:

* Are polygons:

  * more detailed?
  * oversimplified?
  * noisier?

---

## 8. Shift Analysis (Key Insight)

This is your strongest finding.

### What you already showed:

* ~3m average shift
* variable direction
* no global correction possible

### Now extend:

Compare per country:

* mean shift vector (dx, dy)
* variance
* direction distribution

👉 Key question:

> Is SHIFTED systematic per country or random?

---

## 9. Interpretation of SHIFT Error

Very important:

> SHIFTED is the most frequent error category, but is largely attributable to misalignment between Open Buildings and underlying imagery rather than algorithmic failure.

Then:

* show that postprocessing often produces **visually correct alignment**
* but is penalized because reference is shifted

👉 This is a **core limitation of your evaluation**

---


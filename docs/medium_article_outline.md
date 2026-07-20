# Medium article outline: cropweed-seg

Working notes for the article. Each section has the point it must land, suggested
phrasings in my voice, and marked spots for assets.

Legend:
- `[IMAGE: ...]` a figure to insert, with what it should show
- `[CODE: ...]` a short code or terminal snippet
- `[REF: ...]` a link or citation to add
- `[DATA: ...]` a table or number to pull from the repo

---

## Title candidates

Pick one, keep the others as subtitles or section names:

1. "I spent three months finding weeds in peanut fields. The hard part wasn't the model."
2. "Weed IoU 0.670: how I found the ceiling of a segmentation task and proved it was real"
3. "From 400 field photos to 72 fps on a Jetson: an end-to-end segmentation project"

Lean toward 1 or 2. The ceiling story and the free-compression story are the two
angles a Medium reader hasn't seen a hundred times. "I trained U-Net on a Kaggle
dataset" is the article everyone has already read; this one needs to signal early
that it goes somewhere else.

---

## 1. The hook (2-3 short paragraphs)

The point: a reader who knows nothing about agriculture or segmentation should
understand the problem and the punchline in 30 seconds.

Draft direction:

> Peanut yields drop sharply when weeds compete with the crop early in the
> season. A robot that can tell weed from crop from soil, in real time, on
> cheap hardware, is a real precision-agriculture product. This article is
> about building that model end to end: the data, the training, the wall I hit
> at weed IoU 0.67, and how the whole thing ended up running at 72 fps on a
> Jetson board the size of a sandwich.

Then the honest framing sentence, early:

> The interesting part isn't the architecture. It's that the hardest class
> stalled at 0.67 no matter what I tried, and figuring out *why* turned out to
> be the real project.

- `[IMAGE: one strong field photo with its mask overlaid, side by side. Generate
  with scripts/make_demo_frames.py. This is the cover image, pick the prettiest.]`
- `[REF: peanut dataset repo, https://github.com/ptdkhoa/Peanut-dataset, and the
  paper: Tran & Phan, IEEE Access 2023, CC BY-SA 4.0]`

---

## 2. The data, and the two imbalances (the setup section)

The point: 3.7% weed pixels shapes every later decision. Get the reader to feel
the imbalance before showing any model.

Key facts to state plainly:
- 400 RGB images, 960x720, fields near Da Nang, Vietnam.
- Three classes. Background 84.0% of pixels, crop 12.3%, weed 3.7%.
- The twist worth a full paragraph: weed is rare per pixel but present in
  every image; crop is common per pixel but absent from 29.5% of images. Two
  opposite imbalances, two different fixes.

The lazy-model thought experiment goes here, it's the best teaching device in
the article:

> A model that never predicts weed at all still gets 96.3% of pixels right.
> Useless for the actual job, and almost top marks. Every choice that follows,
> the loss, the metric, the split, exists to make that lazy model lose.

Data hygiene paragraph (short, but it builds trust):
- Masks verified pixel-perfect: exactly three clean colors across all 276M
  pixels, so RGB-to-index conversion is lossless. Verify before trusting, a
  principle that returns in the deployment section.
- Split stratified by crop presence (the only presence that varies), 70/15/15,
  seeded, written to versioned text files so every experiment saw identical
  splits.

- `[IMAGE: class balance bar chart, or a grid of 3-4 sample images showing how
  thin and scattered the weed regions are]`
- `[CODE: the has_crop stratification snippet from scripts/prepare_data.py,
  about 10 lines. Shows the whole idea in one screen.]`
- `[DATA: exact split sizes from data/splits/*.txt]`

---

## 3. The loss study (the method section)

The point: focal+dice beats weighted cross-entropy by +0.174 weed IoU, and the
reader should understand *why*, not just see the table.

Structure inside the section:

a) Why weighted CE fails. It fights imbalance with a blunt instrument: every
   weed pixel counts 25x more, whether the model finds it easy or hopeless,
   and it still grades pixel by pixel, blind to shape. Result: weed 0.496.

b) Focal in one idea: questions you already ace count less. Easy background
   fades out of the loss automatically and the class imbalance gets handled as
   a side effect of difficulty weighting. Nobody told the loss which class is
   rare.

c) Dice in one idea: grade the shape, not the pixels. Overlap between predicted
   and true weed region, normalized per class, so a billion correct background
   pixels can't buy back an ignored weed class. The lazy model scores exactly
   zero. Weakness: noisy on tiny regions (small denominators).

d) The combo: pixel-level signal plus region-level signal, failures that cover
   each other. 0.670, reproduced exactly across two seeds.

- `[DATA: the four-row loss table from README / ADR 0003]`
- `[IMAGE: per-epoch weed IoU curves for the four losses, from runs/<name>/
  metrics CSVs. The dice instability dip to 0.45 is visible and makes the
  "noisy on small regions" claim concrete.]`
- `[REF: segmentation-models-pytorch; focal loss paper (Lin et al., 2017) if
  citing sources]`

The war story, its own subsection, maybe titled "The comparison that almost
lied to me":

> At 15 epochs, focal and focal+dice looked tied and I nearly picked plain
> focal for simplicity. Neither had converged. At 25 epochs a real gap opened:
> 0.650 vs 0.670. Comparing methods at a fixed budget is only fair if every
> method has finished converging; otherwise you're measuring which one is
> faster out of the gate, not which one is better.

- `[IMAGE: the two curves crossing / diverging between epoch 15 and 25, with a
  vertical line at 15 marking where the wrong conclusion lived]`

Also state the evaluation choice here: per-class IoU from a confusion matrix
accumulated over the whole split, never mIoU alone. My own numbers are the
argument: 0.971 / 0.910 / 0.670 average to a handsome 0.850 that hides exactly
the class the business problem is about.

---

## 4. The ceiling (the centerpiece section)

The point: three failed improvements make the story stronger, not weaker. This
is the section that separates the article from the usual training write-up.

The elimination logic, spelled out:
- A skeptic has three comebacks to "0.67 is the ceiling": wrong architecture,
  too little resolution, suboptimal loss.
- DeepLabV3+ landed at 0.655. Comeback one dead. 720 crops landed at 0.659.
  Comeback two dead. The four-way loss study killed comeback three.
- Three independent knobs, and the score refuses to move on any of them. If the
  wall were an artifact of one choice, changing that choice would move it.

The line to build the section around:

> Negative results are usually worthless individually and valuable in bundles.
> One failed experiment is an anecdote. Three failed experiments along
> independent axes, converging on the same number, is a measurement of the
> ceiling.

External corroboration paragraph (the finishing move):
- PSPEdgeWeedNet (Pai et al., Sci Rep 2025), with an edge-aware architecture
  and CRF post-processing, still reports weed as its worst class at roughly
  0.60 to 0.69 on the same dataset, and names the small single-region dataset
  as the key limitation. Independent team, heavier machinery, same wall.

Error analysis paragraph, which explains *where* the ceiling lives:
- The worst cases share one pattern: thin, filamentous, low-contrast weed gets
  missed. Errors are false negatives on fine structures, not class confusion.
- So the diagnosis is data, not model: 400 images from one region, and stems
  genuinely ambiguous at this resolution. The only lever that plausibly breaks
  0.67 is more data, and more diverse data.

- `[IMAGE: docs/img/error_gallery.png, already exists. Worst cases by weed IoU
  with error maps. This figure carries the whole diagnosis.]`
- `[IMAGE: optional, the "three levers converge" diagram from the README,
  re-rendered as a clean figure]`
- `[DATA: DeepLabV3+ 0.655 and 720-crop 0.659 numbers, ADR 0004]`
- `[REF: PSPEdgeWeedNet, Pai et al., Scientific Reports 2025]`

---

## 5. Compression for free (the quantization section)

The point: cutting the model 4x cost nothing measurable, and there's a real
explanation, not luck.

Structure inside the section:

a) Why quantize at all: 97.7 MB of 32-bit weights is a lot for a board in a
   field. INT8 stores each number in 8 bits: 24.6 MB, 3.98x smaller.

b) Calibration in plain language. The dial-with-256-ticks analogy worked well
   when I explained this out loud, keep it:

   > A dial with 256 tick marks is fine if you set its endpoints well. If real
   > values run from 10 to 40 and the dial runs 10 to 40, readings stay sharp.
   > Set it from -200 to +500 just in case, and everything bunches up on a few
   > ticks in the middle. Calibration is feeding 50 photos through the model
   > and watching, just watching, to find each dial's real endpoints.

c) The leak that never happened: the 50 calibration photos come from train
   only. Calibration doesn't change what the model knows, but the endpoints
   become part of the deployed model, and if they were tuned on test photos,
   "zero accuracy loss on unseen data" would quietly become "on data the
   compression was tuned for". Even steps that don't train the model can
   contaminate the exam.

d) Why the accuracy survived, the part readers will actually remember:
   - Classification only needs the *ranking* of the three class scores to
     survive, not the scores themselves. Rounding noise flips a prediction only
     on near-ties, and near-ties were coin flips already. That's why weed IoU
     moved by 0.0001 in both directions: noise, not damage.
   - The MinMax result as evidence: three calibration methods compared, the
     fancy outlier-robust ones couldn't beat plain min-and-max. The insurance
     was never needed, so the activations carry no problematic outliers.

- `[DATA: fp32 vs int8 per-class table, README / ADR 0005]`
- `[CODE: the quantize_static call from scripts/quantize.py, a few lines, shows
  QDQ + MinMax choices]`
- `[REF: ONNX Runtime static quantization docs; ADR 0005 in the repo]`

Bridge sentence before this section, on ONNX export: the model was exported to
ONNX and checked three ways before anything else was built on it: max numerical
difference ~1e-5, 100% prediction agreement, identical test IoU. Same principle
as the mask round-trip. Verify each conversion before trusting it.

---

## 6. On the Jetson (the payoff section)

The point: 72 fps INT8 on a $250 board, with numbers a robotics person would
respect.

Key numbers, stated plainly:
- Engines built on device from the fp32 ONNX. Sizes follow the arithmetic:
  97.7 MB fp32, 49.4 fp16, 25.1 int8.
- FP16: 22.38 ms median, 44.7 fps. INT8: 13.88 ms, 72.0 fps. 1.61x faster.
- INT8 weed IoU on device: 0.6689. Drop of 0.0006, noise territory.

Two explanations to include, both short:

a) Why INT8 is 1.61x faster and not 2x. On a small edge GPU the bottleneck is
   moving data, not computing on it: the chef chops fast but the pantry door
   is narrow. INT8 halves the bytes through the door. It isn't a clean 2x
   because some layers stay in higher precision and fixed per-inference costs
   don't shrink.

b) Why median vs P95 matters: 13.88 vs 13.89 ms. A robot doesn't experience
   your average; you provision a real-time system for its tail. A model that's
   usually 14 ms but sometimes 40 is a 40 ms model for planning purposes. This
   one is a 14 ms model, full stop. Honest caveat: clocks were pinned
   (MAXN_SUPER, jetson_clocks); a battery-powered robot with thermal limits
   might not hold them.

Measurement hygiene sentence, because it earns trust: 50 warmup iterations, 500
timed runs, GPU inference only in the timed region, copies and argmax outside.
Say it explicitly: 72 fps means the model runs at 72 fps, not the whole
camera-to-decision pipeline.

- `[DATA: benchmark table from README / results/benchmark.json]`
- `[IMAGE: photo of the actual Jetson Orin Nano on a desk or in a case. A real
  photo of real hardware does more for credibility than any chart.]`
- `[IMAGE: optional, latency histogram from the 500 runs if the raw data is in
  benchmark.json]`
- `[REF: JetPack 6.2 / TensorRT 10.3; docs/deployment_runbook.md in the repo]`

---

## 7. Closing (short)

Not a summary. Two or three of the transferable lessons, then out:

- Per-class metrics or you're lying to yourself: the average was 0.850 while
  the class that mattered sat at 0.670.
- A fair comparison needs every method converged, not a fixed epoch count.
- Failed experiments in bundles measure ceilings; the diagnosis (data, not
  model) is the useful output.
- Verify every conversion step. The lossless mask check and the ONNX fidelity
  check are the same habit at different ends of the pipeline.

End with what's next, concretely, one line each: cross-dataset generalization
(Bonn sugar beets, needs label remapping), and rebuilding the engines on a
newer JetPack to isolate how much the stack alone improves.

- `[REF: link to the GitHub repo, prominently. The repo is the proof.]`

---

## Asset checklist (prepare before drafting)

| Asset | Source | Status |
| --- | --- | --- |
| Cover image, field photo + mask overlay | scripts/make_demo_frames.py | to generate |
| Class balance / sample grid | notebook 01 | to export |
| Loss curves, 4 losses | runs/<name>/ metrics CSVs | to plot |
| 15 vs 25 epoch convergence figure | same CSVs | to plot |
| Error gallery | docs/img/error_gallery.png | done |
| Three-levers diagram | README mermaid, re-render | to make |
| fp32 vs int8 table | README | done |
| Benchmark table | results/benchmark.json | done |
| Jetson photo | take one | to shoot |

## Reference list (collect links before drafting)

- Peanut dataset: github.com/ptdkhoa/Peanut-dataset (Tran & Phan, IEEE Access
  2023, CC BY-SA 4.0)
- PSPEdgeWeedNet: Pai et al., Scientific Reports 2025
- segmentation-models-pytorch
- Focal loss: Lin et al., ICCV 2017
- ONNX Runtime static quantization docs
- TensorRT / JetPack 6.2 docs
- My repo: github.com/alexmnz29/cropweed-seg

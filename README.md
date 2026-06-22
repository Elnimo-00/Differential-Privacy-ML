# dp-sgd fundamentals

training neural networks that provably forget. differentially private stochastic gradient descent on CIFAR-10, with membership inference attack evaluation.

## what this is

standard neural networks memorize their training data. ask a model a carefully crafted question and you can often tell whether a specific person's record was used to train it. this is a **membership inference attack (MIA)**, and it's a real threat in healthcare, finance, and any domain where the training set is sensitive.

**differential privacy (DP)** is the mathematical framework that makes memorization impossible by design. **DP-SGD** is its practical implementation for deep learning. this notebook walks through the complete pipeline:

- the formal (ε, δ)-DP definition and what it actually guarantees
- how gradient clipping + gaussian noise turns standard SGD into DP-SGD
- how opacus implements this in pytorch (including what breaks and why)
- training three models: no privacy, moderate privacy (ε=10), strong privacy (ε=3)
- running a confidence-threshold MIA on all three and measuring attack advantage
- visualizing the privacy-utility tradeoff across all three regimes

## the math

### (ε, δ)-differential privacy

a randomized algorithm M satisfies (ε, δ)-differential privacy if, for **any two datasets D and D' that differ in exactly one record**, and for any possible output set S:

```
Pr[M(D) ∈ S] ≤ e^ε · Pr[M(D') ∈ S] + δ
```

**ε (epsilon)** is the privacy budget. smaller means stronger privacy. e^ε bounds how much more likely any output is when one person's data is included vs excluded. at ε=1 the bound is 2.7x; at ε=10 it's ~22,000x.

**δ (delta)** is a small failure probability. the guarantee holds with probability at least 1−δ. must satisfy δ < 1/n (here: δ = 1e-5 for 50,000 training samples).

**intuition**: an adversary who observes the fully trained model cannot reliably tell whether Alice was in the training set, because the model's output distribution barely shifts with or without her data.

### how DP-SGD works

standard SGD computes the average gradient over a batch and takes a step. DP-SGD adds three mechanisms on top:

```
for each training step:
  1. compute per-sample gradients g_i for every sample i in the batch
  2. CLIP:   g_i ← g_i / max(1, ‖g_i‖₂ / C)       # bound L2 sensitivity to C
  3. SUM:    G   ← Σ g_i
  4. NOISE:  G   ← G + N(0, σ²C²I)                  # calibrated gaussian noise
  5. UPDATE: θ   ← θ - lr · G / batch_size
```

**why clip first?** the gaussian mechanism requires knowing the sensitivity of the function, meaning how much one record can shift the output. by clipping each per-sample gradient to L2 norm ≤ C, we guarantee that removing one sample shifts the gradient sum by at most C. this lets us calibrate noise precisely.

**gaussian mechanism**: for L2 sensitivity Δf, adding noise σ = Δf · √(2 ln(1.25/δ)) / ε achieves (ε, δ)-DP. in DP-SGD, Δf = C (the clipping threshold).

### privacy composition via RDP

each gradient step consumes a slice of the total privacy budget. opacus tracks this using **rényi differential privacy (RDP)**, a tighter accounting method than naive composition (which would sum ε linearly). RDP lets you train for many steps while keeping the final ε budget low.

key tradeoff: to maintain the same final ε across more steps, each step must inject more noise, which means harder optimization and lower accuracy.

### privacy amplification by subsampling

opacus uses **poisson subsampling** where each sample is independently included in a batch with probability q = batch_size / n. randomly subsampling before each step amplifies the privacy guarantee. an attacker can't even be sure a record participated in a given step, which reduces effective ε.

## architecture constraints

not every pytorch layer works with DP-SGD. the key requirement is that per-sample gradients must be computable independently:

| layer | opacus compatible | reason |
|---|---|---|
| `Conv2d`, `Linear` | yes | per-sample computation is independent |
| `BatchNorm2d` | **no** | computes statistics across the batch and couples sample gradients |
| `GroupNorm`, `LayerNorm`, `InstanceNorm` | yes | normalize within each sample |

`ModuleValidator.fix()` auto-replaces `BatchNorm` with `GroupNorm` where possible. always run it before attaching the `PrivacyEngine`.

the model used here is a 4-layer CNN with groupnorm, no batchnorm, totaling ~660K parameters.

## membership inference attack

the **confidence-threshold MIA** is the simplest and most broadly effective attack:

1. run the model on all training samples (members) and all test samples (non-members)
2. record the max softmax confidence for each
3. sweep thresholds, predict "member" if confidence > threshold
4. pick the threshold that maximizes attack accuracy
5. report **advantage = attack accuracy - 0.5** (0 = no better than random; 0.5 = perfect)

a model that memorizes will output very high confidence on training samples and lower confidence on unseen samples, creating a detectable gap. DP-SGD reduces memorization and collapses that gap.

## requirements

```
torch
torchvision
opacus
safetensors
numpy
matplotlib
```

also requires the htb_ai_library helper:

```bash
pip install git+https://github.com/PandaSt0rm/htb-ai-library
```

## dataset

CIFAR-10 is downloaded automatically by the notebook via torchvision:

```python
get_cifar10_loaders(batch_size=256, download=True)
```

no manual setup needed. it will download to `./data/` on first run.

## usage

```bash
pip install -r requirements.txt
pip install git+https://github.com/PandaSt0rm/htb-ai-library
jupyter notebook dp_sgd_fundamentals.ipynb
```

run all cells in order. the notebook trains three models sequentially:
- baseline (no DP), ~20 epochs
- DP ε=10, ~20 epochs, σ ≈ 1.2
- DP ε=3, ~20 epochs, σ ≈ 3.8

GPU recommended. on CPU expect around 10-20 minutes per model. outputs are saved to `output/` and `figs/`.

## results

| model | test accuracy | mia advantage | overfitting gap | noise σ |
|---|---|---|---|---|
| baseline (no DP) | ~67% | ~0.019 | ~10% | 0 |
| DP ε=10 (moderate) | ~58% | ~0.008 | ~3% | ~1.2 |
| DP ε=3 (strong) | ~53% | ~0.004 | ~1% | ~3.8 |

the baseline's 10% overfitting gap is the fingerprint of memorization. the model knows its training set better than the real distribution. under DP ε=3, that collapses to 1%.

MIA advantage drops from 0.019 to 0.004 at ε=3. the attacker is nearly indistinguishable from random guessing.

the accuracy cost at ε=3 is ~14 percentage points. the model is still useful, but there's a real price for strong privacy.

weakness: CIFAR-10 is a relatively easy benchmark. the accuracy gap widens on harder tasks or smaller datasets where the noise-to-signal ratio becomes more damaging.

## key takeaways

DP provides a worst-case upper bound that holds against any attack including ones not yet invented. the empirical MIA gives a current lower bound that measures one known attack today. together they bound the real privacy risk from both sides.

when not to use DP-SGD: very small datasets (noise drowns signal), ε < 1 without public pre-training (accuracy collapses), or when the threat model concerns aggregate statistics rather than individual records.

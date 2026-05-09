# arbitrary depth Stokes waves

This folder contains the public release version of the training and testing
code for the arbitrary depth Stokes-wave MSN and FNO models.

Files:

- `MSN.py`: three-stage multi-stage multi-scale network for predicting Fourier coefficients and the Bernoulli constant.
- `FNO.py`: Fourier neural operator for inverse conformal mapping.

This subfolder is intended for a code-only public upload. Model weights can be released after paper acceptance.

## Requirements

Python packages:

```bash
pip install numpy torch tqdm
```

`tqdm` is optional. CUDA is used automatically when available.

## Input Data Format

Both scripts expect coefficient files named like:

```text
r0_0.70_r_6000_stp_0.033472_Iter_1_O_1000_c0_1.000_a.txt
```

The filename provides:

- `r0`: depth-related parameter.
- `r`: number of Fourier coefficients in the file.
- `stp`: wave steepness parameter.

The file body should contain `r + 1` numeric values:

```text
a_1
a_2
...
a_r
K
```

Lines beginning with `#` are ignored. The final value is interpreted as the
Bernoulli constant `K`.

## MSN Usage

Train the full three-stage MSN:

```bash
python MSN.py train \
  --data-dir data/coefficients \
  --out-dir runs/msn
```

Useful options:

- `--no-train-stage1`, `--no-train-gr`, `--no-train-stage2`: reuse existing
  checkpoints for selected stages.
- `--s1-epochs`, `--gr-epochs`, `--s2-epochs`: training epochs.
- `--device cuda`, `--device cpu`, or `--device auto`.

MSN checkpoint layout:

```text
runs/msn/
  stage1/
    best_Scale1_DNN_*.pth
    ...
    best_B_MLP_*.pth
  global_refiner/
    best_global_refiner.pth
  stage2/
    best_s2_s1.pth
    ...
    best_s2_B.pth
```

Evaluate a saved MSN checkpoint root:

```bash
python MSN.py test \
  --data-dir data/coefficients \
  --ckpt-root runs/msn \
  --out-dir eval_msn
```

Outputs:

- `eval_msn/msn_test_summary.csv`: overall and per-slice MSE.
- `eval_msn/sample_XXXXX_compare.csv`: one sample's coefficient comparison.

Predict coefficient files using saved `.pth` files:

```bash
python MSN.py predict \
  --input-dir data/to_predict \
  --ckpt-root runs/msn \
  --out-dir predictions/msn
```

Each output file is named `pred_<original_filename>` and contains `r + 1`
values: predicted `a_1 ... a_r` followed by predicted `K`.

## FNO Usage

Train the FNO:

```bash
python FNO.py train \
  --train-coeffs-dir data/train \
  --val-coeffs-dir data/val \
  --out-dir runs/fno \
  --cache-dir runs/fno_cache
```

If `--val-coeffs-dir` is omitted, the script randomly splits `--coeffs-dir`
using `--train-split`.

FNO checkpoint:

```text
runs/fno/best_fno_thetaR.pt
```

Test a trained FNO checkpoint:

```bash
python FNO.py test \
  --test-dir data/test \
  --ckpt runs/fno/best_fno_thetaR.pt \
  --out-dir eval_fno
```

Outputs:

- `eval_fno/fno_test_summary.csv`: per-case errors for `theta`, `R`, velocity
  components, and velocity magnitude.
- `eval_fno/dat/`: Tecplot `.dat` files for ground truth, prediction, and
  absolute error fields. Disable this with `--no-write-dat`.

## Checkpoint Usage

After the model files are released, they can be used as follows:

```text
checkpoints/
  msn/
    stage1/
    global_refiner/
    stage2/
  fno/
    best_fno_thetaR.pt
```

Then run:

```bash
python MSN.py predict --input-dir data/to_predict --ckpt-root checkpoints/msn
python FNO.py test --test-dir data/test --ckpt checkpoints/fno/best_fno_thetaR.pt
```

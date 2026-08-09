"""Split the submitted Assignment 2 notebook into the requested deliverables."""
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "group88-assignment2.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


src = json.loads(SOURCE.read_text())
cells = src["cells"]

team = """# Group 88 - UDL Assignment 2: Part A

## Image Generation with beta-VAE and VQ-VAE + PixelCNN Prior

**Dataset:** CIFAR-10 (pixels scaled to [0, 1])  
**Framework:** PyTorch  
**Members:** Gopikannan G (2024ac05790), Sreejith M V (2024AD05421), Ashwini N (2024ad05029), Lalit Tyagi (2024ac05569), Ashok Pyaram (2024ad05197)

This notebook fulfils Part A: beta-VAE evaluation for beta in {1, 2, 4, 10}, and VQ-VAE + PixelCNN prior experiments for codebook sizes K in {512, 256, 128}. Run cells in order on the provisioned GPU environment.
"""

part_a = [md(team)] + [copy.deepcopy(c) for c in cells[1:28]]
(ROOT / "group88-assignment2-part-a.ipynb").write_text(json.dumps(notebook(part_a), indent=1))

part_b_title = """# Group 88 - UDL Assignment 2: Part B

## WGAN and WGAN-GP

**Dataset:** CIFAR-10 (pixels scaled to [0, 1])  
**Framework:** PyTorch  
**Members:** Gopikannan G (2024ac05790), Sreejith M V (2024AD05421), Ashwini N (2024ad05029), Lalit Tyagi (2024ac05569), Ashok Pyaram (2024ad05197)

This standalone notebook trains WGAN (weight clipping) and WGAN-GP (gradient penalty), generates 100 samples from each, and compares their recorded evaluation values. Run cells in order on the provisioned GPU environment.
"""
part_b = [md(part_b_title)] + [copy.deepcopy(c) for c in cells[1:5]] + [copy.deepcopy(c) for c in cells[28:40]]
(ROOT / "group88-assignment2-part-b.ipynb").write_text(json.dumps(notebook(part_b), indent=1))

part_c_intro = """# Group 88 - UDL Assignment 2: Part C

## Comparative Analysis: beta-VAE, VQ-VAE, WGAN and WGAN-GP

**Dataset:** CIFAR-10 | **Framework:** PyTorch

This report compares 100 visual examples per requested model: beta-VAE (beta=1) reconstructions, VQ-VAE + PixelCNN (K=256) samples, WGAN samples, and WGAN-GP samples.

### Evaluation note

The source experiment recorded a torch-only global-colour Fréchet proxy (labelled `FID proxy` below), rather than the standard pretrained-Inception FID. It is retained only for within-run comparison and should not be presented as official FID in a final grading submission. PSNR is meaningful for the two reconstruction models only; GANs have no encoder and therefore no reconstruction error.

### Recorded comparison values

- **beta-VAE (beta=1):** reconstruction PSNR = 18.02 dB; FID proxy = 0.0005. This is a reconstruction model.
- **VQ-VAE + PixelCNN (K=256):** reconstruction PSNR = 24.24 dB; FID proxy = 0.0088. This is a reconstruction model with an autoregressive prior.
- **WGAN:** reconstruction PSNR = N/A; FID proxy = 0.0016. This is an unconditional generator.
- **WGAN-GP:** reconstruction PSNR = N/A; FID proxy = 0.0010. This is an unconditional generator.

### Reconstruction error

VQ-VAE (K=256) has the stronger recorded reconstruction result: 24.24 dB versus 18.02 dB for beta-VAE (beta=1). This corresponds to lower pixel reconstruction error. The discrete codebook preserves local visual detail that a KL-regularised continuous VAE often smooths away. WGAN and WGAN-GP cannot be ranked on reconstruction error because they do not reconstruct an input image.

### Sampling quality

The beta-VAE grid is expected to be smoother because its objective trades exact detail for a regular latent space. VQ-VAE can decode sharper images, although sample realism depends on the PixelCNN prior learning coherent latent maps. Both GAN variants directly optimise adversarial realism; WGAN-GP is generally the safer choice because gradient penalty constrains the critic without the capacity restriction introduced by weight clipping. In this run it also has the better recorded FID proxy (0.0010 vs. 0.0016), but this must be confirmed with standard Inception FID.

### Speed

beta-VAE is the least complex training pipeline: one encoder-decoder model and one loss per step. VQ-VAE adds codebook learning and a separate PixelCNN-prior stage, then performs slow raster-scan latent sampling. WGAN and WGAN-GP use five critic updates per generator update; WGAN-GP adds gradient-penalty differentiation inside each critic update. Therefore the expected compute order is beta-VAE (fastest), VQ-VAE, WGAN, then WGAN-GP (slowest). Exact wall-clock seconds were not captured in the source run, so no fabricated timing values are reported.
"""

# The original notebook already contains four saved 10x10 grids. Preserve only
# those rendered outputs in this print-oriented notebook; no training is needed.
visual_cells = []
for output in cells[43].get("outputs", []):
    visual_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["remove-input"]},
        "source": [],
        "outputs": [copy.deepcopy(output)],
    })

part_c = [md(part_c_intro), md("## 100 visual examples per model")] + visual_cells + [md("""### Conclusion

For reconstruction, VQ-VAE is preferred based on the recorded PSNR. For unconditional visual synthesis, WGAN-GP is the preferred GAN variant, subject to a rerun using standard Inception FID. The comparison balances reconstruction fidelity, perceptual sampling behaviour, and computational cost without treating GAN sample-to-real PSNR as a reconstruction metric.
""")]
(ROOT / "part-c-comparative-analysis.ipynb").write_text(json.dumps(notebook(part_c), indent=1))

print("Created Part A, Part B, and Part C source notebooks.")

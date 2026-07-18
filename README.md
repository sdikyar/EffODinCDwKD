# EffODinCDwKD
**Official repository for the paper: "Efficient Object Detection in Compressed Domain by Exploiting Knowledge Distillation from Pixel Domain"**

This repository provides the official implementation of a novel dual-phase framework designed to achieve fast and efficient object detection directly within partially decoded compressed-domain video data. 

Traditional object detection architectures rely exclusively on fully decoded pixel-domain inputs, making the decoding phase a severe latency bottleneck for real-time edge analytics. To address this, our framework bypasses full decoding and operates directly on compressed data using two key mechanisms:

*   **Low-Frequency Spectral Prioritization:** A partial decoding paradigm that systematically discards high-frequency residual coefficients in the HEVC pipeline. By retaining only a sparse subset of fundamental spatial frequencies, this method dramatically reduces transmission payloads and accelerates the standard decoding process.
*   **Multi-Granularity Cross-Domain Knowledge Distillation:** An architecture designed to recover the structural fidelity lost from discarding residual data. It aligns global contextual features, foreground boundary attention maps, and final response logits to transfer rich representational capacities from a high-performing pixel-domain teacher to a lightweight compressed-domain student network.

By combining these methods, this framework significantly reduces average decoding latency while outperforming conventional fully decoded pixel-domain baselines.

### Acknowledgments
This repository is build upon [HM-18.0](https://vcgit.hhi.fraunhofer.de/jvet/HM) and [DetKDS](https://github.com/lliai/DetKDS).

### Citation
If you find this repository or our paper useful for your research, please consider citing:

```bibtex
@Article{jimaging12070325,
AUTHOR = {Dikyar, Serhat and Toreyin, Behcet Ugur},
TITLE = {Efficient Object Detection in Compressed Domain by Exploiting Knowledge Distillation from Pixel Domain},
JOURNAL = {Journal of Imaging},
VOLUME = {12},
YEAR = {2026},
NUMBER = {7},
ARTICLE-NUMBER = {325},
URL = {[https://www.mdpi.com/2313-433X/12/7/325](https://www.mdpi.com/2313-433X/12/7/325)},
ISSN = {2313-433X},
DOI = {10.3390/jimaging12070325}
}



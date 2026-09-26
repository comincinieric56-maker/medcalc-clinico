# Open ECG Digitizer attribution

MEDCALC vendors the inference source and pretrained U-Net weights from:

- Project: Open ECG Digitizer
- Upstream repository: https://github.com/Ahus-AIM/Open-ECG-Digitizer
- Pinned source commit: `97a15087d4abcda843da8c58ee74b1d8f47e6f9a`
- Upstream license: Creative Commons Attribution-ShareAlike 4.0 International
- Full license text: `LICENSE_OPEN_ECG_DIGIZER.txt` / `LICENSE_OPEN_ECG_DIGITIZER.txt`

The vendored MEDCALC copy contains two inference-only adaptations:
1. the training-only Ray dependency in `src/utils.py` is optional;
2. the experimental dewarper dependency is not loaded when dewarping is disabled.

These changes do not retrain or alter the pretrained U-Net weights.

Upstream citation requested by the project:

Stenhede E, Bjørnstad AM, Ranjbar A. *Digitizing Paper ECGs at Scale:
An Open-Source Algorithm for Clinical Research*. 2025.
doi:10.48550/arXiv.2510.19590.

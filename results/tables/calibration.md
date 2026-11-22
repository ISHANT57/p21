| Model | Input | Training | 1% | 10% | 100% |
|---|---|---|---|---|---|
| Scratch CNN | RGB (3 bands) | supervised | 18.1 ± 0.8 | 3.0 ± 0.1 | 0.6 (n=1) |
| Scratch CNN | 13 bands | supervised | 16.2 ± 0.8 | 1.9 ± 0.2 | 0.5 (n=1) |
| ResNet-50 | RGB (3 bands) | supervised | 6.2 ± 0.9 | 1.7 ± 0.3 | 0.8 (n=1) |
| ResNet-50 | 13 bands | supervised | 6.6 ± 0.4 | 1.1 ± 0.1 | 0.6 (n=1) |
| ViT-Small | RGB (3 bands) | supervised | 4.9 ± 0.7 | 1.5 ± 0.2 | 0.8 (n=1) |
| ViT-Small | 13 bands | supervised | 3.9 ± 0.9 | 1.3 ± 0.2 | 0.7 (n=1) |
| Scratch CNN | RGB (3 bands) | pseudo-label | 15.8 (n=1) | 2.5 (n=1) | - |
| ResNet-50 | RGB (3 bands) | pseudo-label | 4.0 ± 1.5 | 1.2 ± 0.2 | - |
| ResNet-50 | 13 bands | pseudo-label | 4.5 ± 0.3 | 0.8 ± 0.1 | - |

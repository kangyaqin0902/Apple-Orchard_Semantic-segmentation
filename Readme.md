**Model Implementation**

The experimental framework consists of two parts: baseline comparisons and our proposed method.



**1. Comparative Baseline Models**



The following open-source implementations were used as benchmarks. We thank the authors for their contributions to the community:



U-Net: bigmb/Unet-Pytorch



DeepLabV3+ \& PSPNet: Tramac/awesome-semantic-segmentation-pytorch



SegNet: alexgkendall/SegNet-Tutorial



Swin-Unet: HuCaoFighting/Swin-Unet



**2. Our Proposed Method**



The core components developed in this study are organized into the following modules in the root directory:



CycleGan\_Pytorch\_Apple Orchard: Optimized for cross-sensor (JL1KF01A PMS06 image to Sentinel-2 MSI image) domain adaptation.



Swin-Unet-transLearning: The fine-tuned architecture and transfer learning scripts tailored for orchard mapping.






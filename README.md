## ITE-GOYO

# Members
김태림
이원규
정호영
장준일
---
# Introduction

The primary motivation of this project is to advance Active Noise Control (ANC) systems in smart home by shifting from indiscriminate noise suppression to intelligent, selective cancellation. We aim to efficiently distinguish between disruptive mechanical noises (e.g., vacuums, refrigerator) and essential environmental sounds (e.g., human speech, alarms).
To achieve this, we implemented and comparatively analyzed fxine-tuned models based on YAMNet and PANNs to develop a robust real-time audio classification framework. Our ultimate goal is to implement a highly reliable control system that utilizes distributed reference microphones and spatio-temporal multi-stage filtering to precisely detect target appliance noise and trigger the ANC module only when necessary, thereby optimizing both noise reduction performance and computational efficiency on edge devices.

---
# Description of datasets.

Description of the datasets
We constructed a robust dataset by aggregating high-quality samples from multiple verified open-source libraries to maximize classification accuracy and ensure data diversity. 
Appliance Data Sources:
-	Air Conditioner: Samples were extracted from the UrbanSound8K dataset, a widely used benchmark containing 8,732 labeled sound excerpts from urban environments. We selectively used the relevant class to ensure our model is trained on realistic background noise profiles.
-	Vacuum Cleaner: Data was sourced from ESC-50, a standard collection of 2,000 environmental audio recordings, providing clear and distinct motor sound signatures essential for accurate detection.
-	Microwave, Hair Dryer, & Refrigerator: Due to the scarcity of these specific classes in standard datasets, we collected samples using the Freesound API. We strictly filtered for files with CC0 (Public Domain) or CC-BY (Attribution) licenses to ensure full copyright compliance.

  
The Rejection Class:
-	To prevent false positives in smart home environments, we defined an 'Others' class comprising common ambient sounds that should not trigger the ANC system. This class includes human speech, TV audio, and other frequent non-appliance household noises, collected via Freesound to represent a realistic acoustic backdrop.

---
# Methodology

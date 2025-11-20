# ITE-GOYO

## Members
| Name | Organization | Email |
|------|---------------|-------|
| Taerim Kim | Department of Information Systems, Hanyang University | [trnara5375@gmail.com](mailto:trnara5375@gmail.com) |
| Wongyu Lee | Department of Information Systems, Hanyang University | [onew2370@hanyang.ac.kr](mailto:onew2370@hanyang.ac.kr) |
| Junill Jang | Department of Information Systems, Hanyang University | [jang1161@hanyang.ac.kr](mailto:jang1161@hanyang.ac.kr) |
| Hoyoung Chung | Department of Information Systems, Hanyang University | [sydney010716@gmail.com](mailto:sydney010716@gmail.com) |
## Introduction

The primary motivation of this project is to advance Active Noise Control (ANC) systems in smart home by shifting from indiscriminate noise suppression to intelligent, selective cancellation. We aim to efficiently distinguish between disruptive mechanical noises (e.g., vacuums, refrigerator) and essential environmental sounds (e.g., human speech, alarms).

To achieve this, we implemented and comparatively analyzed fxine-tuned models based on YAMNet and PANNs to develop a robust real-time audio classification framework. Our ultimate goal is to implement a highly reliable control system that utilizes distributed reference microphones and spatio-temporal multi-stage filtering to precisely detect target appliance noise and trigger the ANC module only when necessary, thereby optimizing both noise reduction performance and computational efficiency on edge devices.

## **Description of datasets**

We constructed a robust dataset by aggregating high-quality samples from multiple verified open-source libraries to maximize classification accuracy and ensure data diversity. 

### Appliance Data Sources:
+ **Air Conditioner:** Samples were extracted from the UrbanSound8K dataset, a widely used benchmark containing 8,732 labeled sound excerpts from urban environments. We selectively used the relevant class to ensure our model is trained on realistic background noise profiles.

+ **Vacuum Cleaner:** Data was sourced from ESC-50, a standard collection of 2,000 environmental audio recordings, providing clear and distinct motor sound signatures essential for accurate detection.

+ **Microwave, Hair Dryer, & Refrigerator:** Due to the scarcity of these specific classes in standard datasets, we collected samples using the Freesound API. We strictly filtered for files with CC0 (Public Domain) or CC-BY (Attribution) licenses to ensure full copyright compliance.

  
### **The Rejection Class:**
+ To prevent false positives in smart home environments, we defined an 'Others' class comprising common ambient sounds that should not trigger the ANC system. This class includes human speech, TV audio, and other frequent non-appliance household noises, collected via Freesound to represent a realistic acoustic backdrop.


## Methodology

This section details the algorithmic framework and system architecture designed for precise, real-time classification and control of household appliance noise within resource-constrained edge device environments. To maximize computational efficiency, we adopted a MobileNetV1-based deep learning algorithm. Furthermore, we independently designed and implemented a custom transfer learning strategy and a multi-stage filtering algorithm to overcome data imbalance and real-world environmental variability.

### Core Architecture: Depthwise Separable Convolutions
Inference latency is the most critical factor for real-time ANC control. Consequently, we selected the MobileNetV1 (YAMNet Backbone) architecture as our core classification algorithm, replacing standard CNNs to significantly reduce computational overhead.

+ **Algorithmic Efficiency**: MobileNetV1 utilizes the Depthwise Separable Convolution technique, which decomposes standard convolution operations into distinct Depthwise and Pointwise convolutions. This approach drastically reduces the number of parameters and Multiply-Accumulate operations (MACs) compared to traditional CNNs. This efficiency is decisive in processing 0.5-second real-time audio streams with minimal latency.
+ **Feature Extraction:** We leveraged pre-trained YAMNet weights from TensorFlow Hub as a Feature Extractor. To ensure compatibility with our specific requirements, we wrapped the model to align with our system's input/output specifications.

```python
class YAMNetLayer(tf.keras.layers.Layer):
    """
    (batch_size, 15600) 모양의 원본 오디오(waveform) 배치를 입력받아
    (batch_size, 1, 1024) 모양의 YAMNet 임베딩(embeddings) 배치를 출력
    """
    def __init__(self, **kwargs):
        super(YAMNetLayer, self).__init__(**kwargs)
        self.yamnet_tf_function = hub.load('https://tfhub.dev/google/yamnet/1') # 모델로드
        self.trainable = False # 학습되지 않도록

    def call(self, inputs):
        # 배치 내의 각 샘플(15600,)에 대해 실행할 함수를 정의합니다.
        def run_yamnet_on_sample(waveform_1d):
            outputs_tuple = self.yamnet_tf_function(waveform_1d)
            return outputs_tuple[1] #YAMNet의 출력값에서 1024의 embedding값만 가져옴.

        # map_fn을 사용해 inputs의 모든 항목에 함수를 적용
        batch_embeddings = tf.map_fn(
            fn=run_yamnet_on_sample,
            elems=inputs,
            fn_output_signature=tf.TensorSpec(shape=(1, 1024), dtype=tf.float32)
        )
        return batch_embeddings #최종 결과 (batch_size, 1, 1024) 모양의 텐서를 반환.

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, 1024)
```

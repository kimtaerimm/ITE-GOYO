# ITE-GOYO

## Members
| Name | Organization | Email |
|------|---------------|-------|
| Taerim Kim | Department of Information Systems, Hanyang University | [trnara5375@gmail.com](mailto:trnara5375@gmail.com) |
| Wongyu Lee | Department of Information Systems, Hanyang University | [onew2370@hanyang.ac.kr](mailto:onew2370@hanyang.ac.kr) |
| Junill Jang | Department of Information Systems, Hanyang University | [jang1161@hanyang.ac.kr](mailto:jang1161@hanyang.ac.kr) |
| Hoyoung Chung | Department of Information Systems, Hanyang University | [sydney010716@gmail.com](mailto:sydney010716@gmail.com) |
## I. Introduction

The primary motivation of this project is to advance Active Noise Control (ANC) systems in smart home by shifting from indiscriminate noise suppression to intelligent, selective cancellation. We aim to efficiently distinguish between disruptive mechanical noises (e.g., vacuums, refrigerator) and essential environmental sounds (e.g., human speech, alarms).

To achieve this, we implemented and comparatively analyzed fxine-tuned models based on YAMNet and PANNs to develop a robust real-time audio classification framework. Our ultimate goal is to implement a highly reliable control system that utilizes distributed reference microphones and spatio-temporal multi-stage filtering to precisely detect target appliance noise and trigger the ANC module only when necessary, thereby optimizing both noise reduction performance and computational efficiency on edge devices.

## II. Description of datasets

We constructed a robust dataset by aggregating high-quality samples from multiple verified open-source libraries to maximize classification accuracy and ensure data diversity. 

### Appliance Data Sources:
+ **Air Conditioner:** Samples were extracted from the UrbanSound8K dataset, a widely used benchmark containing 8,732 labeled sound excerpts from urban environments. We selectively used the relevant class to ensure our model is trained on realistic background noise profiles.

+ **Vacuum Cleaner:** Data was sourced from ESC-50, a standard collection of 2,000 environmental audio recordings, providing clear and distinct motor sound signatures essential for accurate detection.

+ **Microwave, Hair Dryer, & Refrigerator:** Due to the scarcity of these specific classes in standard datasets, we collected samples using the Freesound API. We strictly filtered for files with CC0 (Public Domain) or CC-BY (Attribution) licenses to ensure full copyright compliance.

  
### **The Rejection Class:**
+ To prevent false positives in smart home environments, we defined an 'Others' class comprising common ambient sounds that should not trigger the ANC system. This class includes human speech, TV audio, and other frequent non-appliance household noises, collected via Freesound to represent a realistic acoustic backdrop.


## III. Methodology

This section details the algorithmic framework and system architecture designed for precise, real-time classification and control of household appliance noise within resource-constrained edge device environments. To maximize computational efficiency, we adopted a MobileNetV1-based deep learning algorithm. Furthermore, we independently designed and implemented a custom transfer learning strategy and a multi-stage filtering algorithm to overcome data imbalance and real-world environmental variability.

### Core Architecture: Depthwise Separable Convolutions
---
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
        # 배치 내의 각 샘플(15600,)에 대해 실행할 함수를 정의.
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

### Training Strategy: Imbalance Handling & 2-Phase Optimization
---

Data within the specialized domain of household appliance noise is inherently sparse, presenting significant challenges in constructing large-scale datasets. To overcome the limitations associated with small-scale datasets (i.e., data scarcity and class imbalance) and to enhance the model's generalization capabilities, we designed the following tailored training algorithms.

#### A. Class-Aware Augmentation & Weighting

Instead of applying uniform random augmentation, we implemented a conditional augmentation logic that adjusts intensity based on specific class characteristics.
+ Target Classes (Appliances): We applied strong augmentation techniques, such as Pitch Shifting and Noise Injection, to artificially synthesize diversity and mitigate the risk of overfitting due to limited data.
+ Rejection Class (Others): Since this class represents the background environment, we employed conservative augmentation to preserve the original acoustic features.

Simultaneously, we computed class weights inversely proportional to the number of samples (Inverse Frequency Weighting) and applied them to the Cross-Entropy Loss function. This mathematically compensates for the imbalance by ensuring that the model prioritizes learning from underrepresented minority classes.

```python
#Augmentation depending on size of datasets.
import numpy as np
import librosa

def add_noise(audio_data, noise_factor=0.005): #add noise
    noise = np.random.randn(len(audio_data))
    augmented_data = audio_data + noise_factor * noise
    augmented_data = augmented_data.astype(type(audio_data[0]))
    return augmented_data

def pitch_shift(audio_data, sample_rate, n_steps=4): #피치 무작위로
    return librosa.effects.pitch_shift(y=audio_data, sr=sample_rate, n_steps=n_steps)
    
def mask_time(audio_data, t_width_max=1000):
    augmented_data = audio_data.copy()
    num_masks = np.random.randint(1,5)
    for _ in range(num_masks):
        t = np.random.randint(0, augmented_data.shape[0])
        t_width = np.random.randint(1, t_width_max + 1)

        # 오디오 길이를 넘지 않도록 끝부분을 보정
        if t + t_width > augmented_data.shape[0]:
            t_width = augmented_data.shape[0] - t
        augmented_data[t:t+t_width] = 0
    return augmented_data

def mask_freq(wav_data, f_width_max=10):
    stft = librosa.stft(wav_data) # 오디오를 스펙트로그램으로 변환
    f_count_max = stft.shape[0] # 총 주파수 밴드 수
    num_masks = np.random.randint(1, 5) # 마스크 개수 1 ~ 5 랜덤
    
    for _ in range(num_masks):
        f = np.random.randint(0, f_count_max)
        f_width = np.random.randint(1, f_width_max + 1)
    
        if f + f_width > stft.shape[0]: #stft.shape[0]을 넘지 않도록 보정
            f_width = stft.shape[0] - f
        stft[f:f+f_width, :] = 0 # 해당 주파수 밴드를 0으로(무음) 만듦
    return librosa.istft(stft)
```

```python
if self.augment:
  current_class = self.class_names[label] # 현재 데이터의 클래스 이름 확인

  if current_class != 'Others':
      if np.random.rand() > 0.5:
          wav_data = add_noise(wav_data, noise_factor=np.random.uniform(0.001, 0.005))
      if np.random.rand() > 0.3:
          wav_data = pitch_shift(wav_data, self.sample_rate, n_steps=np.random.randint(-2, 3))
      if np.random.rand() > 0.3:
          wav_data = mask_time(wav_data)
      if np.random.rand() > 0.7:
          wav_data = mask_freq(wav_data)

  else:
      if np.random.rand() > 0.8:
          wav_data = add_noise(wav_data, noise_factor=np.random.uniform(0.001, 0.005))
      if np.random.rand() > 0.8:
          wav_data = pitch_shift(wav_data, self.sample_rate, n_steps=np.random.randint(-2, 3))
      if np.random.rand() > 0.8:
          wav_data = mask_time(wav_data)
      if np.random.rand() > 0.8:
          wav_data = mask_freq(wav_data)
```
```python
#weighting depending on size of datasets of each classes.
class_weights = class_weight.compute_class_weight(
    'balanced', #데이터 개수에 반비례하게 가중치를 줌 (적을수록 많은 가중치)
    classes=np.unique(train_labels), # np.unique(y_train) -> train_labels
    y=train_labels                   # y_train -> train_labels
)
class_weight_dict = {i : class_weights[i] for i in range(len(class_weights))}
```

#### B. 2-Phase Fine-Tuning Protocol
To mitigate the risk of 'Catastrophic Forgetting' inherent in transfer learning, we implemented a rigorous two-phase optimization protocol.
+ Phase 1 (Warm-up): The backbone network is frozen, and only the custom classifier head is trained using a relatively high learning rate ($10^{-3}$). This step stabilizes the initial weights of the dense layers before modifying the feature extractor.
+ Phase 2 (Fine-tuning): The backbone is unfrozen to allow end-to-end adaptation. However, we critically ensure that Batch Normalization (BN) layers remain frozen while applying a very low learning rate ($10^{-5}$). This technique is pivotal for adapting the model to the specific appliance domain while preserving the robust statistical feature distributions learned from the large-scale source dataset (AudioSet).
 
```python
#2-Phase Fine-Tuning
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=20, # phase1은 20 에폭만 진행
    class_weight=class_weight_dict,
    validation_data=val_generator,
    callbacks=[checkpoint_cb],
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)

print("[Phase 2](Unfreeze Backbone)")

yamnet_found = False
for layer in model.layers:
    # 레이어 이름에 'yamnet'이 있거나 타입이 YAMNetLayer면 품.
    if 'yamnet' in layer.name.lower() or 'YAMNetLayer' in str(type(layer)):
        layer.trainable = True
        yamnet_found = True
        print(f"-> Unfrozen Layer: {layer.name}")

if not yamnet_found:
    print("error: YAMNet 레이어를 찾지 못했습니다.")
    model.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(
    train_generator,
    initial_epoch=20, # 20부터 이어서 시작
    epochs=100,
    class_weight=class_weight_dict,
    validation_data=val_generator,
    callbacks=[checkpoint_cb, early_stop], # 콜백 추가
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator)
)
```

### Real-time Inference & Control Algorithm
---
Given the presence of diverse non-stationary noises in real-world residential environments, relying solely on the instantaneous softmax probability of a 'single' deep learning model poses a high risk of false positives. To address this, we designed a **Dual-Stage Filtering Pipeline** to mathematically and statistically verify the reliability of inferences.
#### A. Optimization: Decibel-based VAD Gating (Pre-filtering)
To maximize the efficiency of resource-constrained edge devices, we deployed a Db VAD(Voice Activity Detection) algorithm prior to the deep learning inference stage.
+ The input audio stream is monitored in real-time using $0.5s$ chunks, and the average decibel level is calculated for each chunk.
+If the calculated $dB$ value is below a predefined threshold (e.g., $55dB$), the signal is considered insignificant 'background noise' and is immediately dropped. This functions as a Kill Switch, preventing the deep learning model from running during silence, thereby conserving computational resources and power.
#### B. Stability: Spatio-Temporal Consistency Filtering
For valid signals passing the VAD gate, we applied a Spatio-Temporal Consistency Algorithm that integrates spatial location information with temporal continuity to generate the final control signal.
+ Sliding Window Buffering
  + The system utilizes a Sliding Window technique to capture continuous context rather than relying on a single data point.
  + By striding every $0.5s$, it generates $1.0s$ overlapping windows and loads them into a FIFO (First-In-First-Out) queue, constructing a time-series buffer ($N=5$ chunks) representing the most recent 3 seconds.
+ Majority Voting Logic (Temporal Consistency):
  + The five independent audio chunks stored in the buffer are each processed by the deep learning model, returning five predicted classes ($C_{pred}$).
  + The system compares these predictions with the Spatial ID (Target Class, $C_{target}$) assigned to the specific microphone to calculate the number of matches.
  + Finally, a control signal is generated only if the condition "Is the same target noise detected in 4 or more out of 5 independent trials ($\ge 80\%$) within a 3-second window ($T=3s$)?" is met.


> $$Trigger = \begin{cases} \text{True (ON)} & \text{if } \sum_{i=1}^{5} \mathbb{I}(C_{pred}^{(i)} == C_{target}) \ge 4 \\ 
\text{False (OFF)} & \text{otherwise} \end{cases}$$

## IV. Evaluation & Analysis
<img width="790" height="516" alt="image" src="https://github.com/user-attachments/assets/517e1227-3ac0-4d18-aa95-a8b70207f16c" />

## V. Related Work
### Foundational Studies (Theoretical Background)
---
This project is built upon state-of-the-art research in efficient deep learning and audio event classification. We leveraged transfer learning from large-scale pre-trained models to ensure both reliability and real-time performance.
#### MobileNets (Backbone Architecture)
**[1] A. G. Howard et al., "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," arXiv preprint arXiv:1704.04861, 2017.**
+ Relevance: This paper introduces the MobileNetV1 architecture, which utilizes depthwise separable convolutions to drastically reduce computational cost ($MACs$). We cited this work to justify our choice of YAMNet as a lightweight backbone suitable for low-latency inference on edge devices.
#### PANNs (Comparative Benchmark)
**[2] Q. Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 2880-2894, 2020.**
+ Relevance: This study proposes Cnn14 (PANNs), a state-of-the-art audio classification model. We utilized this model as a comparative benchmark to analyze the trade-offs between model size, accuracy, and latency, ultimately confirming that lighter models are more effective for our specific ANC application.
#### AudioSet (Pre-training Dataset)
**[3] J. F. Gemmeke et al., "Audio Set: An ontology and human-labeled dataset for audio events," in Proc. IEEE ICASSP, 2017, pp. 776-780.**
+ Relevance: This paper details the large-scale dataset used to pre-train the YAMNet model. Referencing this validates that our model possesses a robust foundational understanding of general audio features before we applied transfer learning with our custom appliance dataset.


### Implementation Tools & Libraries
---
The system implementation relies on standard open-source libraries for deep learning and audio signal processing to ensure reproducibility and stability.
#### Deep Learning Frameworks
+ **TensorFlow & Keras:** Utilized as the primary framework for building the custom classifier head, managing the 2-Phase Fine-Tuning loop, and executing model optimization for edge deployment.
+ **PyTorch:** Used to implement and evaluate the PANNs (Cnn14) model for performance benchmarking.
#### Audio Processing
+ **Librosa:** Utilized for core audio preprocessing tasks, including loading audio files, resampling to 16kHz, trimming silence ($top\_db=30$), and generating spectrograms for analysis.
+ **SoundDevice:** Integrated for real-time low-latency audio stream acquisition from distributed reference microphones.
#### Data Acquisition
+ **Freesound API:** Used to programmatically crawl and filter high-quality datasets based on specific query tags and CC0/CC-BY licenses to address data scarcity.
## VI. Conclusion
not yet

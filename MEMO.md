Transformer encoder architectures have recently achieved state-of-the-art results on monocular 3D human mesh reconstruction, but they require a substantial number of parameters and expensive computations.
Transformer エンコーダー アーキテクチャは、最近、
単眼3D ヒューマンメッシュ再構成で最先端の結果を達成しましたが、かなりの数のパラメーターと高価な計算が必要です。


Due to the large memory overhead and slow inference speed, it is difficult to deploy such models for practical use.
メモリのオーバーヘッドが大きく、推論速度が遅いため、このようなモデルを実用化するのは困難です。

In this paper, we propose a novel transformer encoder-decoder architecture for 3D human mesh reconstruction from a single image, called FastMETRO.
この論文では、FastMETROと呼ばれる、単一の画像からの3Dヒューマンメッシュ再構成のための新しいトランスエンコーダーデコーダーアーキテクチャを提案します。

We identify the performance bottleneck in the encoder-based transformers is caused by the token design which introduces high complexity interactions among input tokens.
エンコーダーベースのトランスフォーマーのパフォーマンスのボトルネックは、入力トークン間の複雑な相互作用を導入するトークン設計によって引き起こされることを特定しました。

We disentangle the interactions via an encoder-decoder architecture, which allows our model to demand much fewer parameters and shorter inference time.
われわれはエンコーダー/デコーダー アーキテクチャを介して相互作用を解きほぐし、これにより、モデルが要求するパラメーターがはるかに少なくなり、推論時間が短縮されます。

In addition, we impose the prior knowledge of human body's morphological relationship via attention masking and mesh upsampling operations, which leads to faster convergence with higher accuracy.
さらに、アテンション マスキングとメッシュアップサンプリング操作を介して、人体の形態学的関係に関する事前知識を課すことで、より高い精度で収束を高速化します。

Our FastMETRO improves the Pareto-front of accuracy and efficiency, and clearly outperforms image-based methods on Human3.6M and 3DPW.
われわれのFastMETRO は、精度と効率のパレートフロントを改善し、Human3.6M および 3DPW での画像ベースの方法よりも明らかに優れています。

Furthermore, we validate its generalizability on FreiHAND.
さらに、FreiHAND でその一般化可能性を検証します。


################################## 1. introduction

3D human pose and shape estimation models aim to estimate 3D coordinates of human body joints and mesh vertices. These models can be deployed in a wide range of applications that require human behavior understanding, e.g., human motion analysis and human-computer interaction. To utilize such models for practical use, monocular methods [2,9,16,17,21–23,25,26,36,39,43,47] estimate the 3D joints and vertices without using 3D scanners or stereo cameras. This task is essentially challenging due to complex human body articulation, and becomes more difficult by occlusions and depth ambiguity in monocular settings.
To deal with such challenges, state-of-the-art methods [25,26] exploit non-local relations among human body joints and mesh vertices via transformer encoder architectures. This leads to impressive improvements in accuracy by consuming a substantial number of parameters and expensive computations as trade-offs; efficiency is less taken into account, although it is crucial in practice.



We propose FastMETRO which employs a novel transformer encoder-decoder architecture for 3D human mesh recovery from a single image. Our method resolves the performance bottleneck in the encoder-based transformers, and improves the Pareto-front of accuracy and efficiency.
• •





The proposed model converges much faster by reducing optimization difficulty. Our FastMETRO leverages the prior knowledge of human body’s morphological relationship, e.g., masking attentions according to the human mesh topology. We present model-size variants of our FastMETRO. The small variant shows competitive results with much fewer parameters and faster inference speed. The large variant clearly outperforms existing image-based methods on the Human3.6M and 3DPW datasets, which is also more lightweight and faster.


######################################################
3 Method
We propose a novel method, called Fast MEsh TRansfOrmer (FastMETRO). FastMETRO has a transformer encoder-decoder architecture for 3D human mesh recovery from an input image. The overview of our method is shown in Figure 2. The details of our transformer encoder and decoder are illustrated in Figure 3.


3.1 Feature Extractor
Given a single RGB image, our model extracts image features XI ∈ RH×W×C through a CNN backbone, where H × W denotes the spatial dimension size and C denotes the channel dimension size. A 1 × 1 convolution layer takes the image features XI as input, and reduces the channel dimension size to D. Then, a flatten operation produces flattened image features XF ∈ RHW×D. Note that we employ positional encodings for retaining spatial information in our transformer, as illustrated in Figure 3.

# ConvGPT

<img src="convgpt.png" alt="ConvGPT" height="400">

vLLM and SGLang module for ConvGPT.

## Overview
**ConvGPT** introduces a novel approach to Large Language Model compression by integrating 2D convolutional networks directly into the pre-training architecture, rather than relying on post-training quantization or pruning. Designed specifically for **Mobile/Edge (SLM)** use cases, it achieves significant parameter reduction while maintaining high reasoning capabilities.

* **Convolutional Embedding Compression:** Unlike standard Transformers that maintain a constant hidden size throughout, ConvGPT utilizes a **Conv2D + Average Pooling** layer to compress the input hidden state vector by a factor of **9x** before it enters the residual stream. This allows the model to maintain high-dimensional information in the embedding layer and prediction head while operating on a highly efficient, smaller vector in the decoder layers.
* **Causal masking in 2D:** The architecture implements specialized padding and reshaping mechanisms during the convolution steps to strictly preserve autoregressive causality. This eliminates "token leakage" (look-ahead bias), ensuring the model remains robust during generation and prevents the test-time degradation often seen in naive convolutional language models.
* **Extreme Parameter Efficiency:**
* **Current Model:** 164M parameters (comparable performance to a standard 722M parameter architecture) — a **~4.4x size reduction**.
* **Scaling Potential:** The architecture scales efficiently; a configuration with `hidden_size=2048` results in just 266M parameters compared to a 1.7B parameter baseline (a **6.5x reduction**).


* **Performance-to-Size Ratio:** Trained on 250B tokens (PleIAs/SYNTH), this 164M model achieves **>30% on GPQA-Diamond**, a significant outlier for its size class, demonstrating that logic and reasoning capabilities can be preserved even with aggressive vector compression.
* **Normalization Stability:** Includes post-convolution normalization to manage vector value scaling, ensuring training stability and consistent generation output.

## Installation

```sh
pip install -e .
```

## Running

**vLLM**:
```sh
vllm serve mkurman/ConvGPT-SYNTH-250B-EC --served-model-name convgpt --trust_remote_code --max-model-len 16384 --gpu-memory-utilization 0.7 --max_num_batched_tokens 8192
```

**SGLang**:

```sh
export SGLANG_EXTERNAL_MODEL_PACKAGE=convgpt.sglang && python -m sglang.launch_server mkurman/ConvGPT-SYNTH-250B-EC --trust-remote-code --port 8000 --mem-fraction-static 0.7 --context-length 16384 --allow-auto-truncate --attention-backend fa3
```
# GPT-SoVITS-FastInference

> This project is also available in **[Korean](./README_ko.md)**.

A streamlined Python wrapper for fast inference with GPT-SoVITS, based on the fast_inference_ branch of the original GPT-SoVITS project, with significant contributions from ChasonJiang.

This fork is designed solely for inference purposes and does not include the full features present in the original GPT-SoVITS project.

## Introduction

GPT-SoVITS-FastInference offers a simplified wrapper for utilizing the powerful few-shot voice conversion and text-to-speech capabilities of GPT-SoVITS. It's ideal for scenarios where quick and efficient inference is required without the need for a graphical interface.

## Features

- **Efficient Inference**: Benefit from fast inference capabilities optimized for quick processing.
- **Python Wrapper**: Intuitive Python interface for seamless integration into your projects.

## Getting Started

### Installation

To install GPT-SoVITS-FastInference, simply follow the setup instructions below:

- Install Python and run the following commands:

```
pip install git+https://github.com/ohgi07/gpt_sovits_python.git
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# Usage

## Initialize the configuration and specify the path of the pretrained models

```
from gpt_sovits_python import TTS, TTS_Config

soviets_configs = {
    "default_v2": {
        "device": "cuda",  # ["cpu", "cuda"]
        "is_half": True,  # Set 'False' if you will use cpu
        "version": "v2",  # GPT-SoVITS v2
        "t2s_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "vits_weights_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    }
}

tts_config = TTS_Config(soviets_configs)
```

## Load the models

```
tts_pipeline = TTS(tts_config)
```

## Configure the parameters you need

```
params = {
    "text": "",                   # str.(required) text to be synthesized
    "text_lang": "",              # str.(required) language of the text to be synthesized
    "ref_audio_path": "",         # str.(required) reference audio path
    "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
    "prompt_text": "",            # str.(optional) prompt text for the reference audio
    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
    "top_k": 5,                   # int. top k sampling
    "top_p": 1,                   # float. top p sampling
    "temperature": 1,             # float. temperature for sampling
    "text_split_method": "cut0",  # str. text split method, see text_segmentation_method.py for details.
    "batch_size": 1,              # int. batch size for inference
    "batch_threshold": 0.75,      # float. threshold for batch splitting.
    "split_bucket": True,         # bool. whether to split the batch into multiple buckets.
    "return_fragment": False,     # bool. step by step return the audio fragment.
    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
    "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
    "seed": -1,                   # int. random seed for reproducibility.
    "parallel_infer": True,       # bool. whether to use parallel inference.
    "repetition_penalty": 1.35    # float. repetition penalty for T2S model.
    }
```

## Perform inference

```
tts_generator = tts_pipeline.run(params)

sr, audio_data = next(tts_generator)
```
The output is a NumPy array, and you can save or play with it in a notebook.

## Save

```
from scipy.io import wavfile

wavfile.write("output.wav", rate=sr, data=audio_data)
```

# License
This project is licensed under the MIT License, same as the original GPT-SoVITS project.

# Disclaimer
This software is provided for educational and research purposes only. The authors and contributors of this project do not endorse or encourage any misuse or unethical use of this software. Any use of this software for purposes other than those intended is solely at the user's own risk. The authors and contributors shall not be held responsible for any damages or liabilities arising from the use of this software inappropriately.

# Acknowledgments
Original GPT-SoVITS project contributors
ChasonJiang for significant code contributions
davidbrowne17 with code modification to support GPT-SoVITS v2


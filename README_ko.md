# GPT-SoVITS-FastInference (한국어)

> 🇰🇷 이 문서는 **한국어 README**입니다.
> English version is available in [README.md](./README.md).

**GPT-SoVITS-FastInference**는 GPT-SoVITS 원본 프로젝트의 `fast_inference_` 브랜치를 기반으로, **빠른 추론**(Inference)만을 목적에 맞게 단순화한 파이썬 래퍼입니다.  

이 포크는 **GPT‑SoVITS **v2** 모델**까지 지원되며, 학습(Training)·GUI 등 원본 프로젝트의 모든 기능을 포함하지 않습니다.

## 개요

GPT-SoVITS-FastInference는 GPT-SoVITS의 강력한 몇 샷 음성 변환 및 텍스트 음성 변환 기능을 활용하기 위한 간소화된 래퍼를 제공합니다.
그래픽 인터페이스 없이 빠르고 효율적인 추론이 필요한 시나리오에 이상적입니다.

## 특징

- **효율적인 추론**: 빠른 처리에 최적화된 빠른 추론 기능의 이점을 활용하세요.
- **Python 래퍼**: 프로젝트에 원활하게 통합할 수 있는 직관적인 Python 인터페이스.

## 시작하기

### 설치

GPT-SoVITS-FastInference를 설치하려면 아래 설정 지침을 따르기만 하면 됩니다:

- Python을 설치하고 다음 명령을 실행합니다:

```
pip install git+https://github.com/ohgi07/gpt_sovits_python.git
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# 사용법

## 설정 및 사전학습 모델 경로 지정

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

## 모델 로드

```
tts_pipeline = TTS(tts_config)
```

## 필요한 매개변수 구성

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

## 추론 수행

```
tts_generator = tts_pipeline.run(params)

sr, audio_data = next(tts_generator)
```
출력은 NumPy 배열이며, 저장하거나 따로 사용할 수 있습니다.

## 저장

```
from scipy.io import wavfile

wavfile.write("output.wav", rate=sr, data=audio_data)
```

# 라이선스
이 프로젝트는 원래 GPT-SoVITS 프로젝트와 동일하게 MIT 라이선스 하에 라이선스됩니다.

# 면책 조항
이 소프트웨어는 교육 및 연구 목적으로만 제공됩니다. 이 프로젝트의 저자와 기여자는 이 소프트웨어의 오용 또는 비윤리적 사용을 지지하거나 권장하지 않습니다. 의도한 목적 이외의 용도로 이 소프트웨어를 사용하는 것은 전적으로 사용자의 위험에 따릅니다. 저자와 기여자는 부적절하게 이 소프트웨어를 사용하여 발생한 손해나 책임에 대해 책임을 지지 않습니다.

# 감사의 말
GPT-SoVITS 원본 프로젝트 모든 기여자
대규모 코드 개선에 기여한 ChasonJiang
GPT-SoVITS v2 지원을 위해 코드 수정을 진행한 davidbrowne17


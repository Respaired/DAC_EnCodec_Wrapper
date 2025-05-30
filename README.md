# HuggingFace's EnCodec-like Wrapper for Descript Audio Codec
Descript Audio Codec wrapper that makes it identical to how EnCodec is used.
The purpose is to make it an easy switch from EnCodec to DAC without touching anything else (inshallah).

### Install

```bash

git clone https://github.com/Respaired/DAC_HF_Wrapper.git
cd DAC_HF_Wrapper
pip install -e .

```


### Usage

```python

from transformers import AutoProcessor
import librosa
from dac_encodec import DACModel
import torch

from transformers.models.dac.configuration_dac import DacConfig as LibraryDacConfig


device = 'cuda'

original_frame_rate_getter = LibraryDacConfig.frame_rate.fget
def library_dac_config_frame_rate_setter(self_instance, value):
    pass

# bypassing the frame setter error
LibraryDacConfig.frame_rate = property(fget=original_frame_rate_getter,
                                        fset=library_dac_config_frame_rate_setter)


model = DACModel.from_pretrained("parler-tts/dac_44khZ_8kbps").to(device)
processor = AutoProcessor.from_pretrained("parler-tts/dac_44khZ_8kbps", sampling_rate=44_100)

wav, sr = librosa.load('audio.wav', sr=44_100)

inputs = processor(raw_audio=wav,
                   sampling_rate=processor.sampling_rate,
                   return_tensors="pt").to(device)

with torch.no_grad():
    z = model.encode(inputs['input_values'], padding_mask=inputs['padding_mask'])
    output = model.decode(z.audio_codes, audio_scales=None, padding_mask=inputs['padding_mask']).audio_values.detach().cpu().numpy().squeeze()
```


references: <br>
[parler-tts/dac_44khZ_8kbps](https://huggingface.co/parler-tts/dac_44khZ_8kbps)  <br>
[facebook/encodec_24khz](https://huggingface.co/facebook/encodec_24khz)

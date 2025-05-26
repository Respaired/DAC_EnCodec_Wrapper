# DAC_HF_Wrapper
an accurate, honest to God, Descript Audio Codec wrapper for HF that's identical to how EnCodec is used.
The purpose is to make it an easy switch from EnCodec to dac without touching anything else (inshallah).
# Usage

```python

from transformers  AutoProcessor
import librosa
from DAC_HF_Wrapper import DACModel


model = DACModel.from_pretrained("parler-tts/dac_44khZ_8kbps").to('cuda')
processor = AutoProcessor.from_pretrained("parler-tts/dac_44khZ_8kbps")

wav, sr = librosa.load('audio.wav', sr=44_100)



inputs = processor(raw_audio=wav,
 sampling_rate=processor.sampling_rate,
 return_tensors="pt")

z = model.encode(inputs['input_values'], padding_mask=inputs['padding_mask'])
output = model.decode(z.audio_codes, audio_scales=None, padding_mask=inputs['padding_mask']).audio_values.detach().cpu().numpy().squeeze()
```


references: <br>
[parler-tts/dac_44khZ_8kbps](https://huggingface.co/parler-tts/dac_44khZ_8kbps)

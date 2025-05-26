import torch
from dac.model import DAC
from torch import nn
from typing import Optional, Tuple, List 

from transformers import PreTrainedModel
from transformers.models.encodec.modeling_encodec import EncodecDecoderOutput, EncodecEncoderOutput
from transformers.utils import ModelOutput

from .configuration_dac import DACConfig


class DACModel(PreTrainedModel):
    config_class = DACConfig
    main_input_name = "input_values"

    def __init__(self, config: DACConfig):
        super().__init__(config)
        self.config = config


        dac_init_kwargs = {
            "n_codebooks": config.num_codebooks,
            "latent_dim": config.latent_dim,
            "codebook_size": config.codebook_size,
        }
        if hasattr(config, "audio_channels"):
            dac_init_kwargs["channels"] = config.audio_channels


        self.model = DAC(**dac_init_kwargs)
        
        self.remove_weight_norm()
        self.apply_weight_norm()

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        bandwidth: Optional[float] = None, # Not used by DAC, kept for API compatibility
        return_dict: Optional[bool] = None,
        n_quantizers: Optional[int] = None,
        sample_rate: Optional[int] = None,
    ):
        """
        Encodes the input audio waveform into discrete codes.
        """
        batch_size, num_input_channels, input_length = input_values.shape

        if num_input_channels < 1:
            raise ValueError(f"Number of audio input_values channels must be at least 1, but got {num_input_channels}")

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # --- Masking and Preprocessing ---
        if padding_mask is None:
            effective_padding_mask = torch.ones(
                batch_size, input_length, device=input_values.device, dtype=torch.bool
            )
        else:
            effective_padding_mask = padding_mask.bool()

        broadcastable_mask = effective_padding_mask
        if broadcastable_mask.ndim == input_values.ndim - 1:
            broadcastable_mask = broadcastable_mask.unsqueeze(1)
        
        input_values_masked = input_values * broadcastable_mask.to(input_values.dtype)
        

        audio_data_for_encode = self.model.preprocess(input_values_masked, sample_rate=sample_rate)
        

        _, codes, _, _, _ = self.model.encode(audio_data_for_encode, n_quantizers=n_quantizers)
        
        audio_codes_stacked = codes.unsqueeze(0) 
        audio_scales_stacked = [None] 

        if not return_dict:
            return (audio_codes_stacked, audio_scales_stacked)

        return EncodecEncoderOutput(audio_codes=audio_codes_stacked, audio_scales=audio_scales_stacked)

    def decode(
        self,
        audio_codes: torch.Tensor, 
        audio_scales: Optional[list] = None, 
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if audio_codes.shape[0] != 1:
            raise ValueError(
                f"Expected one frame/chunk in audio_codes (shape[0]), got {audio_codes.shape[0]}"
            )
        
        current_audio_codes = audio_codes.squeeze(0)
        latents, _, _ = self.model.quantizer.from_codes(current_audio_codes)
        audio_values = self.model.decode(latents)

        if padding_mask is not None:
            target_len = padding_mask.shape[-1]
            current_len = audio_values.shape[-1]



            if current_len > target_len:
                audio_values = audio_values[..., :target_len]



        if not return_dict:
            return (audio_values,)
        
        return EncodecDecoderOutput(audio_values=audio_values)

    def forward(self, input_values, padding_mask=None, **kwargs):
        encode_outputs = self.encode(
            input_values,
            padding_mask=padding_mask,
            return_dict=True,
            sample_rate=kwargs.get("sample_rate", self.config.sampling_rate),
            n_quantizers=kwargs.get("n_quantizers", None)
        )
        audio_codes = encode_outputs.audio_codes
        audio_scales = encode_outputs.audio_scales

        decode_outputs = self.decode(
            audio_codes=audio_codes,
            audio_scales=audio_scales,
            padding_mask=padding_mask,
            return_dict=True
        )
        reconstructed_audio_values = decode_outputs.audio_values
        

        class _LocalEncodecOutput(ModelOutput): 
            audio_codes: torch.Tensor = None
            audio_values: Optional[torch.Tensor] = None
        
        if kwargs.get("return_dict", self.config.return_dict if hasattr(self.config, "return_dict") else True):
             return _LocalEncodecOutput(audio_codes=audio_codes, audio_values=reconstructed_audio_values)
        
        return (audio_codes, reconstructed_audio_values)

    def apply_weight_norm(self):
        try:
            weight_norm_fn = nn.utils.parametrizations.weight_norm
        except AttributeError:
            weight_norm_fn = nn.utils.weight_norm

        def _apply(module):
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                if not hasattr(module, 'weight_g') and not hasattr(module, 'weight_v'):
                    try:
                        weight_norm_fn(module)
                    except Exception:
                        pass 
        self.model.apply(_apply)

    def remove_weight_norm(self):
        def _remove(module):
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                try:
                    nn.utils.remove_weight_norm(module)
                except ValueError:
                    pass 
                except Exception:
                    pass
        self.model.apply(_remove)


if "EncodecOutput" not in globals():
    class EncodecOutput(ModelOutput):
        audio_codes: torch.Tensor = None
        audio_values: Optional[torch.Tensor] = None

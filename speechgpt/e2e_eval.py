# e2e_eval.py

import torch
import os
import tempfile
import torchaudio
from cli_infer import SpeechGPTInference   # adjust import to your repo


# Global model instance (singleton pattern)
_infer = None

def load_inference_model(
    model_name_or_path: str,
    lora_weights: str = None,
    s2u_dir: str = "utils/speech2unit/",
    vocoder_dir: str = "utils/vocoder/",
    output_dir: str = "output/",
):
    """Load the SpeechGPT inference model once."""
    global _infer
    if _infer is None:
        os.makedirs(output_dir, exist_ok=True)
        _infer = SpeechGPTInference(
            model_name_or_path=model_name_or_path,
            lora_weights=lora_weights,
            s2u_dir=s2u_dir,
            vocoder_dir=vocoder_dir,
            output_dir=output_dir,
        )
    return _infer

import tempfile
import torchaudio
import os

def tensor_to_units(input_audio, sample_rate, infer):
    """
    Convert a waveform tensor to discrete units using Speech2Unit.
    """
    tmp_path = None
    # Ensure shape (channels, time)
    if input_audio.ndim == 1:
        input_audio = input_audio.unsqueeze(0)  # (1, T)
    elif input_audio.ndim == 2 and input_audio.shape[0] > input_audio.shape[1]:
        input_audio = input_audio.T  # fix accidental (T, 1)
    try:
        # create a temp wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            torchaudio.save(tmp_wav.name, input_audio.cpu(), sample_rate)
            tmp_path = tmp_wav.name

        # pass file path to s2u
        units = infer.s2u(tmp_path, merged=True)
        return units
    except Exception as e:
        print(f"Exception{e}")

    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.remove(tmp_path)


def e2e_evaluation(input_audio: torch.Tensor, sample_rate: int):
    """
    End-to-end evaluation for SpeechGPT without temp files.
    """
    import numpy as np

    assert isinstance(input_audio, torch.Tensor), "Input must be a torch.Tensor"
    if input_audio.ndim == 2 and input_audio.size(0) == 1:
        input_audio = input_audio.squeeze(0)
    elif input_audio.ndim > 2:
        raise ValueError("Input audio must be 1D or 2D with shape (1, T)")

    infer = _infer
    if infer is None:
        raise RuntimeError("Inference model is not loaded. Call load_inference_model() first.")

    # print("Inferning model")
    # print("[DEBUG] infer.s2u =", infer.s2u)
    # print("[DEBUG] callable?", callable(infer.s2u))

    # 🔑 Directly pass waveform into Speech2Unit
    # units = infer.s2u(input_audio, merged=True)

    units = tensor_to_units(input_audio, sample_rate, infer)
    # print("Units extracted:", units)

    # Build in-memory prompt
    prompt = f"is input: {units}"

    # Run inference
    generated_sr, generated_audio = infer([prompt])

    # 🚨 Debug: check what type we actually got back
    # print(f"[DEBUG] infer returned type={type(generated_audio)}, shape={getattr(generated_audio, 'shape', None)}")

    # ✅ normalize return to numpy float32
    if isinstance(generated_audio, torch.Tensor):
        generated_audio = generated_audio.squeeze().cpu().numpy()
    generated_audio = np.asarray(generated_audio, dtype=np.float32)

    return generated_sr, generated_audio


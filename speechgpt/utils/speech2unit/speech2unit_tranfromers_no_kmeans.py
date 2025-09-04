import logging
import os
import sys
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample

from transformers import HubertModel

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("generate_embeddings")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FeatureReader(object):
    """Feature extractor using 🤗 Transformers' HubertModel."""

    def __init__(
        self,
        model_id: str = "facebook/hubert-base-ls960",
        layer: int = -1,   # -1 = last_hidden_state
        max_chunk: int = 1600000,
        fp16: bool = False,
        sampling_rate: int = 16000,
        normalize_input: bool = True,
        trust_remote_code: bool = False,
        device: str = "cpu",
    ):
        self.device = device
        self.model = HubertModel.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        self.model.eval().to(self.device)
        self.model.config.output_hidden_states = True

        self.layer = layer
        self.max_chunk = max_chunk
        self.fp16 = fp16 and self.device.startswith("cuda")
        self.target_sample_hz = sampling_rate
        self.normalize_input = normalize_input

        if self.fp16:
            self.model.half()

        logger.info(
            f"Loaded Hubert model '{model_id}' on {self.device} | fp16={self.fp16} | target_sr={self.target_sample_hz}"
        )

    def read_audio(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if sr != self.target_sample_hz:
            wav = resample(wav, sr, self.target_sample_hz)
        # make mono if needed
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav

    @torch.no_grad()
    def get_feats(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.fp16:
            x = waveform.half().to(self.device)
        else:
            x = waveform.float().to(self.device)

        # Optional input normalization
        if self.normalize_input:
            x = F.layer_norm(x, x.shape)

        # flatten to [B=1, T]
        x = x.view(1, -1)

        feats = []
        for start in range(0, x.size(1), self.max_chunk):
            x_chunk = x[:, start : start + self.max_chunk]
            out = self.model(x_chunk, output_hidden_states=True)

            if self.layer == -1:
                feat_chunk = out.last_hidden_state.squeeze(0)  # [T', C]
            else:
                hidden_states = out.hidden_states
                if self.layer >= len(hidden_states):
                    raise ValueError(
                        f"Requested layer {self.layer} but model returned {len(hidden_states)} hidden states."
                    )
                feat_chunk = hidden_states[self.layer].squeeze(0)  # [T', C]

            feats.append(feat_chunk)

        if len(feats) == 0:
            return torch.zeros(0, 0, device=self.device)
        return torch.cat(feats, dim=0)  # [Frames, Dim]


class Speech2Embedding(torch.nn.Module):
    """Speech to continuous embeddings (no k-means)."""

    def __init__(
        self,
        model_id: str,
        layer: int = -1,
        max_chunk: int = 1600000,
        fp16: bool = False,
        sampling_rate: int = 16000,
        normalize_input: bool = True,
        trust_remote_code: bool = False,
        device="cpu"
    ):
        super().__init__()
        self.feature_reader = FeatureReader(
            model_id=model_id,
            layer=layer,
            max_chunk=max_chunk,
            fp16=fp16,
            sampling_rate=sampling_rate,
            normalize_input=normalize_input,
            trust_remote_code=trust_remote_code,
            device=device,
        )

    def __call__(self, path: str) -> torch.Tensor:
        waveform = self.feature_reader.read_audio(path).to(self.feature_reader.device)
        feat = self.feature_reader.get_feats(waveform)
        return feat  # [Frames, Dim]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/hubert-base-ls960",
        help="Hugging Face model id or local path (e.g., 'lengyue233/content-vec-best').",
    )
    parser.add_argument("--layer", type=int, default=-1, help="Layer index, -1 = last_hidden_state.")
    parser.add_argument("--wav", type=str, required=True, help="Path to input wav file.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--max_chunk", type=int, default=1600000)
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device: cpu or cuda")

    args = parser.parse_args()

    s2e = Speech2Embedding(
        model_id=args.model_id,
        layer=args.layer,
        max_chunk=args.max_chunk,
        fp16=args.fp16,
        sampling_rate=args.sampling_rate,
        normalize_input=not args.no_norm,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
    )

    feats = s2e(args.wav)
    print(f"Extracted embeddings shape: {feats.shape}")  # [Frames, Dim]
    np.save("output_embeddings.npy", feats.cpu().numpy())

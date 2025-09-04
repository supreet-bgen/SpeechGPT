import logging
import os
import sys
import joblib
import argparse
import numpy as np
from types import SimpleNamespace

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
logger = logging.getLogger("generate_pseudo_language")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FeatureReader(object):
    """Feature extractor using 🤗 Transformers' HubertModel.

    This mirrors the API of the earlier fairseq-based FeatureReader but loads
    the encoder with `transformers` and fetches hidden states.
    """

    def __init__(
        self,
        model_id: str = "facebook/hubert-base-ls960",
        layer: int = 11,
        max_chunk: int = 1600000,
        fp16: bool = False,
        sampling_rate: int = 16000,
        normalize_input: bool = True,
        trust_remote_code: bool = False,
        device: str = "cpu",   # <--- new
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

        # Optional input normalization similar to some fairseq recipes
        if self.normalize_input:
            x = F.layer_norm(x, x.shape)

        # flatten to [B=1, T]
        x = x.view(1, -1)

        feats = []
        for start in range(0, x.size(1), self.max_chunk):
            x_chunk = x[:, start : start + self.max_chunk]
            # transformers HubertModel expects `input_values`
            out = self.model(x_chunk, output_hidden_states=True)
            hidden_states = out.hidden_states  # tuple

            # In HubertModel (base, 12 transformer layers), hidden_states indices:
            # 0: features from feature extractor (conv), then 1..12: transformer layer outputs
            # Align behavior with fairseq: "layer" typically refers to transformer index.
            # So layer=11 matches the 11th transformer block -> hidden_states[11]
            idx = self.layer
            if idx >= len(hidden_states):
                raise ValueError(
                    f"Requested layer {idx} but model returned {len(hidden_states)} hidden states."
                )

            feat_chunk = hidden_states[idx].squeeze(0)  # [T', C]
            feats.append(feat_chunk)

        if len(feats) == 0:
            return torch.zeros(0, 0, device=self.device)
        return torch.cat(feats, dim=0)


class ApplyKmeans(object):
    def __init__(self, km_path: str):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            self.C = self.C.to(x)
            self.Cnorm = self.Cnorm.to(x)
            dist = x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (x**2).sum(1, keepdims=True) - 2 * np.matmul(x, self.C_np) + self.Cnorm_np
            return np.argmin(dist, axis=1)


class Speech2Unit(torch.nn.Module):
    def __init__(
        self,
        model_id: str,
        km_path: str,
        layer: int = 11,
        max_chunk: int = 1600000,
        fp16: bool = False,
        sampling_rate: int = 16000,
        normalize_input: bool = True,
        trust_remote_code: bool = False,
        device="cpu"
    ):
        """Speech to unit sequence using k-means over Hubert features (HF).

        IMPORTANT: The k-means checkpoint (km_path) MUST have been trained on
        the same encoder *and layer* specified here; otherwise the units will
        be meaningless.
        """
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
        self.apply_kmeans = ApplyKmeans(km_path)

    @staticmethod
    def merge_duplicates(cluster_ids):
        dup_cluster_list = []
        duration_list = []
        count = 1
        for i in range(0, len(cluster_ids)):
            if i + 1 < len(cluster_ids) and cluster_ids[i] == cluster_ids[i + 1]:
                count += 1
            else:
                dup_cluster_list.append(cluster_ids[i])
                duration_list.append(count)
                count = 1
        return dup_cluster_list, duration_list

    def __call__(self, path, merged: bool = True):
        waveform = self.feature_reader.read_audio(path).to(self.feature_reader.device)
        feat = self.feature_reader.get_feats(waveform)
        cluster_ids = self.apply_kmeans(feat).tolist()
        dup_cluster_list, duration_list = self.merge_duplicates(cluster_ids)

        merged_units = "<sosp>" + "".join([f"<{str(x)}>" for x in dup_cluster_list]) + "<eosp>"
        unmerged_units = "<sosp>" + "".join([f"<{str(x)}>" for x in cluster_ids]) + "<eosp>"

        if merged:
            return merged_units
        else:
            return unmerged_units
        # Optionally return continuous feats and durations if needed.


def _infer_km_from_dir(ckpt_dir: str, default_layer: int) -> str:
    """Helper to keep backward compatibility with a directory-based API.

    Will try to pick a km file that matches the layer number if present.
    """
    if not ckpt_dir:
        raise ValueError("ckpt_dir is empty; provide --km_path explicitly.")
    # Try typical pattern: *L{layer}_km*.bin
    candidates = []
    try:
        import glob

        pat = os.path.join(ckpt_dir, f"*L{default_layer}_km*.bin")
        candidates = glob.glob(pat)
        if not candidates:
            # fallback: any .bin
            candidates = glob.glob(os.path.join(ckpt_dir, "*.bin"))
    except Exception:
        pass
    if not candidates:
        raise FileNotFoundError(
            f"Could not find any k-means .bin in {ckpt_dir}. Pass --km_path explicitly."
        )
    logger.info(f"Auto-selected km model: {candidates[0]}")
    return candidates[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Primary HF params
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/hubert-base-ls960",
        help="Hugging Face model id or local path (e.g., 'lengyue233/content-vec-best').",
    )
    parser.add_argument(
        "--km_path",
        type=str,
        default="",
        help="Path to k-means model (.bin/.joblib) trained on the SAME encoder & layer.",
    )
    parser.add_argument("--layer", type=int, default=11, help="Hidden state layer index to use.")
    parser.add_argument("--wav", type=str, required=True, help="Path to input wav file.")

    # Convenience/compatibility
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="",
        help="(Optional) Directory containing km model to infer if --km_path not given.",
    )

    # Runtime knobs
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--max_chunk", type=int, default=1600000)
    parser.add_argument("--no_norm", action="store_true", help="Disable input layer-norm.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow custom model code when loading from Hugging Face.",
    )
    parser.add_argument("--no_merge", action="store_true", help="Do not merge consecutive duplicates.")

    args = parser.parse_args()

    if not args.km_path:
        if not args.ckpt_dir:
            raise SystemExit("Please provide --km_path or --ckpt_dir to locate a k-means model.")
        args.km_path = _infer_km_from_dir(args.ckpt_dir, args.layer)

    s2u = Speech2Unit(
        model_id=args.model_id,
        km_path=args.km_path,
        layer=args.layer,
        max_chunk=args.max_chunk,
        fp16=args.fp16,
        sampling_rate=args.sampling_rate,
        normalize_input=not args.no_norm,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
    )

    units = s2u(args.wav, merged=not args.no_merge)
    print(units)

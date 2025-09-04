import logging
import os
import sys
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample

# from transformers import HubertModel
import joblib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("generate_units")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



class FeatureReader(object):
    def __init__(
        self,
        model_path: str,
        layer: int = 11,
        max_chunk: int = 1600000,
        fp16: bool = False,
        sampling_rate: int = 16000,
        normalize_input: bool = True,
        device: str = "cpu",
    ):
        self.device = device
        self.layer = layer
        self.max_chunk = max_chunk
        self.fp16 = fp16 and self.device.startswith("cuda")
        self.target_sample_hz = sampling_rate
        self.normalize_input = normalize_input

        logger.info(f"Loading Hubert checkpoint from {model_path}")
        # ✅ load fairseq-style checkpoint directly
        from fairseq import checkpoint_utils

        try:
            models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
                [model_path],
                arg_overrides={"data": "."},  # no real data needed
                strict=False
            )
            self.model = models[0].to(self.device)
            self.model.eval()
        except Exception as e:
            raise ValueError(f"Failed to load Hubert checkpoint from {model_path}: {e}")


        self.model.eval().to(self.device)
        if self.fp16:
            self.model.half()

        logger.info(
            f"Loaded Hubert checkpoint '{model_path}' on {self.device} "
            f"| fp16={self.fp16} | target_sr={self.target_sample_hz}"
        )

    def read_audio(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if sr != self.target_sample_hz:
            wav = resample(wav, sr, self.target_sample_hz)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav

    @torch.no_grad()
    def get_feats(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.fp16:
            x = waveform.half().to(self.device)
        else:
            x = waveform.float().to(self.device)

        if self.normalize_input:
            x = F.layer_norm(x, x.shape)

        x = x.view(1, -1)

        feats = []
        for start in range(0, x.size(1), self.max_chunk):
            x_chunk = x[:, start:start + self.max_chunk]
            # ✅ fairseq Hubert forward returns features list
            feat, _ = self.model.extract_features(x_chunk, output_layer=self.layer)
            feats.append(feat.squeeze(0))

        if len(feats) == 0:
            return torch.zeros(0, 0, device=self.device)
        return torch.cat(feats, dim=0)


class ApplyKmeans(object):
    def __init__(self, km_path: str, device="cpu"):
        km_model = joblib.load(km_path)
        C_np = km_model.cluster_centers_.astype(np.float32).T
        Cnorm_np = (C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(C_np).to(device)
        self.Cnorm = torch.from_numpy(Cnorm_np).to(device)

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        x = x.to(self.C.device)
        dist = x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
        return dist.argmin(dim=1).cpu().numpy()


class Speech2Unit(torch.nn.Module):
    def __init__(self, model_path: str, km_path: str, layer: int = 11, device="cpu", **kwargs):
        super().__init__()
        self.feature_reader = FeatureReader(model_path=model_path, layer=layer, device=device, **kwargs)
        self.apply_kmeans = ApplyKmeans(km_path, device=device)

    def merge_duplicates(self, cluster_ids):
        dup_cluster_list = []
        duration_list = []
        count = 1
        for i in range(len(cluster_ids)):
            if i + 1 < len(cluster_ids) and cluster_ids[i] == cluster_ids[i + 1]:
                count += 1
            else:
                dup_cluster_list.append(cluster_ids[i])
                duration_list.append(count)
                count = 1
        return dup_cluster_list, duration_list

    def __call__(self, path, merged=True):
        waveform = self.feature_reader.read_audio(path).to(self.feature_reader.device)
        feat = self.feature_reader.get_feats(waveform)
        cluster_ids = self.apply_kmeans(feat).tolist()
        dup_cluster_list, duration_list = self.merge_duplicates(cluster_ids)

        merged_units = "<sosp>" + "".join([f"<{x}>" for x in dup_cluster_list]) + "<eosp>"
        unmerged_units = "<sosp>" + "".join([f"<{x}>" for x in cluster_ids]) + "<eosp>"

        return merged_units if merged else unmerged_units


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to Hubert .pt checkpoint")
    parser.add_argument("--km_path", type=str, required=True, help="Path to kmeans .bin file")
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--max_chunk", type=int, default=1600000)
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_merge", action="store_true")
    parser.add_argument("--device", type=str, default=DEVICE)

    args = parser.parse_args()

    s2u = Speech2Unit(
        model_path=args.model_path,
        km_path=args.km_path,
        layer=args.layer,
        max_chunk=args.max_chunk,
        fp16=args.fp16,
        sampling_rate=args.sampling_rate,
        normalize_input=not args.no_norm,
        # trust_remote_code=args.trust_remote_code,
        device=args.device,
    )

    units = s2u(args.wav, merged=not args.no_merge)
    print(units)

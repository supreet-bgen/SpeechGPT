import os
import csv
import argparse
import ray
from filelock import FileLock
from tqdm import tqdm

from speech2unit_tranfromers import Speech2Unit  # your existing module


# -------- Persistent GPU Actor --------
@ray.remote(num_gpus=1)
class Speech2UnitActor:
    def __init__(self, model_id, km_path, layer, max_chunk, fp16, sampling_rate, trust_remote_code):
        print(f"[GPU Actor] Loading model {model_id} on GPU...")
        self.s2u = Speech2Unit(
            model_id=model_id,
            km_path=km_path,
            layer=layer,
            max_chunk=max_chunk,
            fp16=fp16,
            sampling_rate=sampling_rate,
            trust_remote_code=trust_remote_code,
        )
        print("[GPU Actor] Model ready.")

    def process(self, wav_path, out_csv, flush_every=50):
        # Run inference on GPU
        units = self.s2u(wav_path, merged=True)
        filename = os.path.basename(wav_path)
        row = (filename, " ".join(units))

        # Safe write with file lock
        lock = FileLock(out_csv + ".lock")
        with lock:
            file_exists = os.path.isfile(out_csv)
            with open(out_csv, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["filename", "units"])
                writer.writerow(row)

        return row


# -------- Main driver --------
def main(args):
    ray.init(ignore_reinit_error=True)

    # Collect wav files
    wav_files = []
    for root, _, files in os.walk(args.wav_root):
        for f in files:
            if f.endswith(".wav"):
                wav_files.append(os.path.join(root, f))
    wav_files.sort()

    print(f"Found {len(wav_files)} wav files.")

    # Spawn one actor per GPU
    actors = [
        Speech2UnitActor.remote(
            model_id=args.model_id,
            km_path=args.km_path,
            layer=args.layer,
            max_chunk=args.max_chunk,
            fp16=args.fp16,
            sampling_rate=args.sampling_rate,
            trust_remote_code=args.trust_remote_code,
        )
        for _ in range(args.num_gpus)
    ]

    # Distribute wavs round-robin across actors
    futures = []
    for i, wav in enumerate(wav_files):
        actor = actors[i % len(actors)]
        futures.append(actor.process.remote(wav, args.out_csv))

    # Progress bar
    for _ in tqdm(ray.get(futures), total=len(futures)):
        pass

    print(f"✅ Done. Results saved to {args.out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_root", type=str, required=True, help="Path to directory containing wav files")
    parser.add_argument("--km_path", type=str, required=True, help="Path to k-means .bin file")
    parser.add_argument("--model_id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--out_csv", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")

    # Optional Hubert params
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--max_chunk", type=int, default=1600000)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--trust_remote_code", action="store_true")

    args = parser.parse_args()
    main(args)
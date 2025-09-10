import torch
import torch.nn as nn
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
import soundfile as sf
from typing import List
import argparse
import logging
import json
from tqdm import tqdm
import os
import re
import sys
import traceback
from peft import PeftModel
import transformers
from transformers import AutoConfig, LlamaForCausalLM, LlamaTokenizer, GenerationConfig

# NEW: import speech2unit_transformers
from utils.speech2unit.speech2unit_tranfromers import Speech2Unit

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NAME = "SpeechGPT"
META_INSTRUCTION = (
    "You are an AI assistant whose name is SpeechGPT.\n"
    "- SpeechGPT is a intrinsic cross-modal conversational language model that is developed by Fudan University. "
    "SpeechGPT can understand and communicate fluently with human through speech or text chosen by the user.\n"
    "- It can perceive cross-modal inputs and generate cross-modal outputs.\n"
)
DEFAULT_GEN_PARAMS = {
    "max_new_tokens": 1024,
    "min_new_tokens": 10,
    "temperature": 0.8,
    "do_sample": True,
    "top_k": 60,
    "top_p": 0.8,
}
device = torch.device("cuda")


def extract_text_between_tags(text, tag1="[SpeechGPT] :", tag2="<eoa>"):
    pattern = f"{re.escape(tag1)}(.*?){re.escape(tag2)}"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else ""


class SpeechGPTInference:
    def __init__(
        self,
        model_name_or_path: str,
        lora_weights: str = None,
        s2u_ckpt: str = "mhubert_base_vp_en_es_fr_it3_L11_km1000.bin",
        s2u_model_id: str = "utter-project/mHuBERT-147",
        vocoder_dir: str = "speechgpt/utils/vocoder/",
        output_dir="speechgpt/output/",
    ):
        self.meta_instruction = META_INSTRUCTION
        self.template = "[Human]: {question} <eoh>. [SpeechGPT]: "

        # ✅ speech2unit (transformers)
        self.s2u = Speech2Unit(
            km_path=s2u_ckpt,
            model_id=s2u_model_id,
            device=device,
        )

        # ✅ load LLM
        self.model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        if lora_weights is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        self.model.half()
        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        # tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

        # generation config
        self.generate_kwargs = DEFAULT_GEN_PARAMS

        # vocoder
        vocoder = os.path.join(vocoder_dir, "vocoder.pt")
        vocoder_cfg = os.path.join(vocoder_dir, "config.json")
        with open(vocoder_cfg) as f:
            vocoder_cfg = json.load(f)
        self.vocoder = CodeHiFiGANVocoder(vocoder, vocoder_cfg).to(device)

        self.output_dir = output_dir

    def preprocess(self, raw_text: str):
        processed_parts = []
        for part in raw_text.split("is input:"):
            path = part.strip()
            print(os.path.splitext(path)[-1])
            print(os.path.isfile(path))
            print(path)
            if os.path.isfile(path) and os.path.splitext(path)[-1] in [".wav", ".flac", ".mp4"]:
                # ✅ extract units with transformers, same as first script
                processed_parts.append(self.s2u(path, merged=True))
                print("Processing Audio Input")
            else:
                processed_parts.append(part)
        processed_text = "is input:".join(processed_parts)
        prompt_seq = self.meta_instruction + self.template.format(question=processed_text)
        return prompt_seq


    def postprocess(self, response: str):
        question = extract_text_between_tags(response, tag1="[Human]", tag2="<eoh>")
        answer = extract_text_between_tags(response + "<eoa>", tag1="[SpeechGPT] :", tag2="<eoa>")
        tq = extract_text_between_tags(response, tag1="[SpeechGPT] :", tag2="; [ta]") if "[ta]" in response else ""
        ta = extract_text_between_tags(response, tag1="[ta]", tag2="; [ua]") if "[ta]" in response else ""
        ua = extract_text_between_tags(response + "<eoa>", tag1="[ua]", tag2="<eoa>") if "[ua]" in response else ""
        return {"question": question, "answer": answer, "textQuestion": tq, "textAnswer": ta, "unitAnswer": ua}

    def forward(self, prompts: List[str]):
        with torch.no_grad():
            preprocessed_prompts = [self.preprocess(p) for p in prompts]

            input_ids = self.tokenizer(preprocessed_prompts, return_tensors="pt", padding=True).input_ids.to(device)

            generation_config = GenerationConfig(
                temperature=0.7,
                top_p=0.8,
                top_k=50,
                do_sample=True,
                max_new_tokens=2048,
                min_new_tokens=10,
            )

            generated_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            responses = self.tokenizer.batch_decode(generated_ids.sequences.cpu(), skip_special_tokens=True)

            responses = [self.postprocess(x) for x in responses]

            os.makedirs(f"{self.output_dir}/wav/", exist_ok=True)
            out_json = f"{self.output_dir}/responses.json"
            init_num = sum(1 for _ in open(out_json)) if os.path.exists(out_json) else 0

            with open(out_json, "a") as f:
                for i, r in enumerate(responses):
                    json_line = json.dumps(r)
                    f.write(json_line + "\n")
                    if r["answer"] and "<sosp>" in r["answer"]:
                        unit = [int(num) for num in re.findall(r"<(\d+)>", r["answer"])]
                        x = {"code": torch.LongTensor(unit).view(1, -1).to(device)}
                        wav = self.vocoder(x, True)
                        self.dump_wav(init_num + i, wav, prefix="answer")
                        print(f"Speech response saved in {self.output_dir}/wav/answer_{init_num+i}.wav")

        return 16000, wav.detach().cpu().numpy() if "wav" in locals() else None

    def dump_wav(self, sample_id, pred_wav, prefix):
        sf.write(f"{self.output_dir}/wav/{prefix}_{sample_id}.wav", pred_wav.detach().cpu().numpy(), 16000)

    def __call__(self, input):
        return self.forward(input)

    def interact(self):
        prompt = str(input(f"Please talk with {NAME}:\n"))
        while prompt != "quit":
            try:
                self.forward([prompt])
            except Exception as e:
                traceback.print_exc()
                print(e)
            prompt = str(input(f"Please input prompts for {NAME}:\n"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, default="")
    parser.add_argument("--lora-weights", type=str, default=None)
    parser.add_argument("--s2u-ckpt", type=str, default="mhubert_base_vp_en_es_fr_it3_L11_km1000.bin")
    parser.add_argument("--s2u-model-id", type=str, default="utter-project/mHuBERT-147")
    parser.add_argument("--vocoder-dir", type=str, default="speechgpt/utils/vocoder/")
    parser.add_argument("--output-dir", type=str, default="speechgpt/output/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    infer = SpeechGPTInference(
        args.model_name_or_path,
        args.lora_weights,
        args.s2u_ckpt,
        args.s2u_model_id,
        args.vocoder_dir,
        args.output_dir,
    )

    infer.interact()

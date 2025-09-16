#stage1: modality-adaptation pretraining
import random
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from datasets import Features, Value, Sequence
import evaluate
import math
import tqdm
import glob
import transformers
from transformers import Trainer, Gemma3ForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint
from utils.prompter import Prompter
import os
from itertools import chain
import logging
import sys
import torch.distributed as dist


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the tokenized data"},
    )

    #tune embedding
    train_embeddings: bool = field(
        default=False,
        metadata={
                "help": (
                    "only train embeddings while training"
                    )
        },
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default="", 
        metadata={"help": "Path to the training data."})
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The input training data file (a text file)."}
        )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    validation_files: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Optional list of validation files for multiple evals."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    use_text: bool = field(
        default=False, metadata={"help": "Use text data for pretraining"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    report_to: Optional[str] = field(
        default="wandb",
        metadata={"help": "The integration to report the results and logs to. Options: ['all', 'wandb', 'tensorboard', 'comet_ml', 'mlflow', 'azureml', 'clearml', 'neptune', 'wandb', 'offline']"}
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the W&B run. If not set, HF generates one automatically."}
    )


import multiprocessing
multiprocessing.set_start_method("forkserver", force=True)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
# ensure each process uses its GPU
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class AggregatingTrainer(Trainer):
    def log(self, logs, start_time=None):
        # Collect all split-specific validation losses
        val_losses = [v for k, v in logs.items() if k.startswith("eval_") and k.endswith("_loss")]
        if val_losses:
            # Average across all validation splits
            logs["eval_loss"] = sum(val_losses) / len(val_losses)
        super().log(logs, start_time)

if dist.is_available() and dist.is_initialized():
    dist.destroy_process_group()


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.world_size > 1}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print(f"##### Training from last checkpoint {last_checkpoint} #####")
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    model = Gemma3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    )#.to(torch.device(training_args.device))

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    # tokenizer.pad_token_id = (
    #     0  # unk. we want this to be different from the eos token
    # )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    tokenizer.padding_side = "left"  # Allow batched inference
    #Extend vocab for speech units
    # Only expand vocab if starting from scratch (no checkpoint)
    # if last_checkpoint is None and training_args.resume_from_checkpoint is None:
    if '<sosp>' not in tokenizer.get_vocab():
        units_size = 1000
        logger.info(f"Add special unit tokens <0>-<{units_size-1} to tokenizer.vocab")
        new_tokens = [f"<{x}>" for x in range(units_size)] + ['<sosp>', '<eosp>']
        tokenizer.add_tokens(new_tokens)

    # resize embedding
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))


    if model_args.train_embeddings:
        logger.info("only update embedding parameters")
        for name, param in model.named_parameters():
            if "embed" not in name:
                param.requires_grad = False


    #data
    
    features = Features({
        "text": Value("string"),
        "tokens": Sequence(Value("string")),    # ✅ instead of List
        "labels": Sequence(Value("int32"))
    })
    data_files = {}
    dataset_args = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.validation_files is not None:
        for i, path in enumerate(data_args.validation_files):
            data_files[f"validation{i}"] = path
    extension = (
        data_args.train_file.split(".")[-1]
        if data_args.train_file is not None
        else data_args.validation_file.split(".")[-1]
    )
    if extension == "txt":
        extension = "text"
    raw_datasets = load_dataset(
        extension,
        cache_dir=None,
        keep_in_memory=False,
        data_files=data_files,
        **dataset_args,
    )
    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if not any(k.startswith("validation") for k in raw_datasets.keys()):
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            features=features,
            cache_dir=None,
            keep_in_memory=False,
            split=f"train[{data_args.validation_split_percentage}%:]",
            **dataset_args,
        )
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            features=features,
            cache_dir=None,
            keep_in_memory=False,
            split=f"train[:{data_args.validation_split_percentage}%]",
            **dataset_args,
        )

    
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        return output
    # >>> ADD CLEANUP HERE <<<
    
    if training_args.local_rank == 0 and training_args.overwrite_output_dir:
        for p in glob.glob(os.path.join(training_args.output_dir, "cache-*.arrow")) + \
                 glob.glob(os.path.join(training_args.output_dir, "grouped-*.arrow")):
            try:
                os.remove(p)
            except OSError:
                pass

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    # <<< END CLEANUP >>>

    tokenized_cache_file_names = {
        "train":os.path.join(model_args.cache_dir, 'tokenized', 'train', 'processed_train.arrow'),
        "validation":os.path.join(model_args.cache_dir, 'tokenized', 'valid', 'processed_valid.arrow'),
        }
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_cache_file_names = {
        split: os.path.join(training_args.output_dir, f"cache-{split}.arrow")
        for split in raw_datasets.keys()
        }
        # tokenization: only rank0 writes cache files; others keep in memory
        if not data_args.streaming:
            if training_args.local_rank == 0:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    batch_size=10000,                        # speedup if memory allows
                    num_proc=max(1, min(128, data_args.preprocessing_num_workers or os.cpu_count())),
                    remove_columns=column_names,
                    load_from_cache_file=True,
                    desc="Running tokenizer on dataset",
                    cache_file_names=tokenized_cache_file_names,
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    batch_size=10000,
                    num_proc=max(1, min(128, (data_args.preprocessing_num_workers or os.cpu_count())//2)),
                    remove_columns=column_names,
                    load_from_cache_file=True,
                    desc="Running tokenizer on dataset",
                    keep_in_memory=True,
                )
        else:
            # streaming path (no in-memory caching)
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )


    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


    group_cache_file_names = {
        "train":os.path.join(model_args.cache_dir, 'group', 'train', 'processed_train.arrow'),
        "validation":os.path.join(model_args.cache_dir, 'group', 'valid', 'processed_valid.arrow'),
        }

    with training_args.main_process_first(desc="grouping texts together"):
        group_cache_file_names = {
            split: os.path.join(training_args.output_dir, f"grouped-{split}.arrow")
            for split in tokenized_datasets.keys()
        }

        if not data_args.streaming:
            if training_args.local_rank == 0:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=min(128, data_args.preprocessing_num_workers or os.cpu_count()),
                    load_from_cache_file=False,
                    desc=f"Grouping texts in chunks of {block_size}",
                    cache_file_names=group_cache_file_names,
                )
            else:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=1,  # 🔑 only 1 worker on non-rank0
                    load_from_cache_file=False,
                    desc=f"Grouping texts in chunks of {block_size}",
                    keep_in_memory=True,
                )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )



    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))            


    if training_args.do_eval:
        eval_datasets = {}

        for key in lm_datasets.keys():
            if key.startswith("validation"):   # catch validation, validation0, validation1, ...
                eval_split = lm_datasets[key]
                if data_args.max_eval_samples is not None:
                    max_eval_samples = min(len(eval_split), data_args.max_eval_samples)
                    eval_split = eval_split.select(range(max_eval_samples))
                eval_datasets[key] = eval_split

        if not eval_datasets:
            raise ValueError("--do_eval requires at least one validation dataset")



    data_collator = DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )

    trainer = AggregatingTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_datasets if training_args.do_eval else None,
        data_collator=data_collator,
    )
    

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

        

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        # metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        total_eval = sum(len(d) for d in eval_datasets.values())
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else total_eval
        )
        metrics["eval_samples"] = min(max_eval_samples, total_eval)

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)



if __name__ == "__main__":
    train()
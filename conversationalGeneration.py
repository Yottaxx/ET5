#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import torch
from datasets import load_dataset, load_metric
from sklearn.metrics import accuracy_score, confusion_matrix

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from trainer import Seq2SeqEncoderParallelTrainer
from utils.collator import DataCollatorForSeq2SeqEntailment
from utils.evaluator import MoreEvaluator, CombinedEvaluator
from utils.modeling_t5 import T5ForConditionalGeneration

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    graphNum: int = field(
        metadata={"help": "whether need encoder classify "}
    )

    dividedLr: float = field(
        metadata={"help": "whether need encoder classify "}
    )

    encoder_classifier: int = field(
        metadata={"help": "whether need encoder classify "}
    )

    encoder_loss: int = field(
        metadata={"help": "whether need encoder classify "}
    )

    classify_only: int = field(
        metadata={"help": "inquire update"}
    )

    decoder_enhance: int = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    loss_entailment: float = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    loss_ce: float = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.entailment_alpha = model_args.loss_entailment
    config.ce_alpha = model_args.loss_ce
    config.decoder_enhance = model_args.decoder_enhance
    config.graphNum = model_args.graphNum
    config.dividedLr = model_args.dividedLr

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    special_tokens_dict = {'additional_special_tokens': ["<edu>", "<final>", "<question>", "<scenario>"]}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if not training_args.do_train and model_args.model_name_or_path not in ['t5-base','t5-large']:
        state_Dict = torch.load("{}/pytorch_model.bin".format(model_args.model_name_or_path),
                                map_location=training_args.device)
        model.load_state_dict(state_dict=state_Dict)

    print(model)

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})

    # model.encoder.embed_tokens = EntailmentEmbeddings(model.encoder.embed_tokens, config)
    # model.encoder.embed_tokens.token_type_embeddings.weight.data.normal_(mean=0.0, std=config.initializer_range)

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )


    # edu version
    # target -> edu answer / final answer
    entailment_list = ['true', 'unknown', 'false']
    prediction_list = ['irrelevant', 'yes', 'no']
    split_str = 'final: '

    def preprocess_function(examples):

        inputs = examples['input']

        relation = examples['relation']

        targets = []
        encoder_label = []
        for i in range(len(examples['target'])):
            temp_target = []

            temp_target.append(examples['target'][i][-1].replace('<edu>', '').replace('<final> ', 'final: ').strip())

            if model_args.classify_only:
                if temp_target[-1].split(split_str)[-1] not in prediction_list:
                    temp_target[-1] = split_str + 'inquire'

            if model_args.encoder_loss:
                if temp_target[-1].split(split_str)[-1] not in prediction_list:
                    encoder_label.append(3)
                else:
                    encoder_label.append(prediction_list.index(temp_target[-1].split(split_str)[-1]))

            targets.append(" ".join(temp_target))

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        edu_entailments_mask = []
        edu_entailments_label = []
        edu_entailment_len = []
        user_idx = []
        rule_idx = []
        scenario = examples['scenario']
        for i in range(len(model_inputs['input_ids'])):

            edu_entailments_mask_temp = (torch.tensor(model_inputs['input_ids'][i]) == tokenizer.convert_tokens_to_ids(
                '<edu>'))

            rule_idx.append(torch.tensor([i for i in range(len(edu_entailments_mask_temp))]).masked_select(
                edu_entailments_mask_temp.bool()))

            final_entailments_mask_temp = (
                    torch.tensor(model_inputs['input_ids'][i]) == tokenizer.convert_tokens_to_ids(
                '<final>'))

            scenario_entailments_mask_temp = (
                    torch.tensor(model_inputs['input_ids'][i]) == tokenizer.convert_tokens_to_ids(
                '<scenario>'))

            assert sum(final_entailments_mask_temp).item() == 1, sum(final_entailments_mask_temp).item()
            assert sum(scenario_entailments_mask_temp).item() <= 1

            question_entailments_mask_temp = (
                    torch.tensor(model_inputs['input_ids'][i]) == tokenizer.convert_tokens_to_ids(
                '<question>'))

            user_idx.append(torch.tensor([i for i in range(len(question_entailments_mask_temp))]).masked_select(
                (final_entailments_mask_temp ^ question_entailments_mask_temp ^ scenario_entailments_mask_temp).bool()))
            assert (
                           edu_entailments_mask_temp ^ final_entailments_mask_temp ^ question_entailments_mask_temp ^ scenario_entailments_mask_temp).sum().item() == \
                   sum(edu_entailments_mask_temp).item() + sum(final_entailments_mask_temp).item() + sum(
                question_entailments_mask_temp).item() + sum(scenario_entailments_mask_temp).item()

            edu_entailments_mask.append(
                (
                        edu_entailments_mask_temp + final_entailments_mask_temp + question_entailments_mask_temp + scenario_entailments_mask_temp
                ).long().tolist()
            )

            temp_entailments_label = []

            assert sum(edu_entailments_mask_temp).item() == len(examples['target'][i]) - 1

            edu_entailment_len.append(sum(edu_entailments_mask_temp).item())

            for j in range(sum(edu_entailments_mask_temp).item()):
                if examples['target'][i][j] == '-100':
                    temp_entailments_label.append(-100)
                else:
                    temp_entailments_label.append(
                        entailment_list.index(
                            examples['target'][i][j].replace("<edu> yes", "<edu> true").replace("<edu> no",
                                                                                                "<edu> false").replace(
                                '<edu> ', '').strip()))

            assert len(temp_entailments_label) == sum(edu_entailments_mask_temp).item()
            edu_entailments_label.append(temp_entailments_label)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["entailment_mask"] = edu_entailments_mask
        model_inputs['entailment_label'] = edu_entailments_label
        model_inputs['encoder_label'] = encoder_label
        model_inputs['entailment_len'] = edu_entailment_len
        model_inputs['relationInput'] = relation
        model_inputs['rule_idx'] = rule_idx
        model_inputs['user_idx'] = user_idx
        model_inputs['scenario'] = scenario
        return model_inputs

    #
    # def preprocess_function(examples):
    #
    #     snippet = examples['snippet']
    #     scenario = examples['scenario']
    #     question = examples['question']
    #     history = examples['history']
    #     predicted = examples['predicted_evidence']
    #     inputs = []
    #
    #     for i in range(len(snippet)):
    #         inputs_item = [snippet[i], scenario[i], question[i]]
    #
    #         for item in history[i]:
    #             inputs_item.append("history: " + item['follow_up_question'])
    #             inputs_item.append(item['follow_up_answer'])
    #
    #         for item in predicted[i]:
    #             inputs_item.append("predicted: " + item['follow_up_question'])
    #             inputs_item.append(item['follow_up_answer'])
    #
    #         inputs.append(tokenizer.eos_token.join(inputs_item))
    #
    #     targets = examples['answer']
    #
    #     inputs = [prefix + inp for inp in inputs]
    #     model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
    #
    #     # Setup the tokenizer for targets
    #     with tokenizer.as_target_tokenizer():
    #         labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
    #
    #     # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    #     # padding in the loss.
    #     if padding == "max_length" and data_args.ignore_pad_token_for_loss:
    #         labels["input_ids"] = [
    #             [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    #         ]
    #
    #     model_inputs["labels"] = labels["input_ids"]
    #     return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2SeqEntailment(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    data_collator.encoder_classifier = model_args.encoder_classifier

    # Metric
    metric = CombinedEvaluator()

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    # def acc_metric(preds, labels):
    #     total_count = 0
    #     right_count = 0
    #     pred_count = 0
    #     for i in range(len(labels)):
    #         if labels[i].strip() != 'none':
    #             total_count += 1
    #             if preds[i].strip() != 'none':
    #                 right_count += 1
    #         if preds[i].strip() != 'none':
    #             pred_count += 1
    #
    #     return right_count / total_count, right_count, total_count, pred_count

    def encoder_acc(preds, labels):
        encoder_len_acc = 0
        total_count = min(len(preds), len(labels))
        total_match = 0
        if len(preds) == len(labels):
            encoder_len_acc = 1

        for i in range(total_count):
            if preds[i] == labels[i]:
                total_match += 1

        return encoder_len_acc, total_match, total_count

    def acc_metric(preds, labels):

        encoder_len_match = 0
        encoder_len_count = 0

        encoder_preds_count = 0
        encoder_label_count = 0

        preds_index = []
        labels_index = []

        for i in range(len(labels)):
            temp_encoder_labels = labels[i].split(split_str)[0].split(' ')
            temp_encoder_preds = preds[i].split(split_str)[0].split(' ')

            if '' in temp_encoder_labels:
                temp_encoder_labels.remove('')
            if '' in temp_encoder_preds:
                temp_encoder_preds.remove('')

            temp_label = labels[i].split(split_str)[-1].strip().lower()
            temp_pred = preds[i].split(split_str)[-1].strip().lower()

            encoder_len_item, encoder_len_match_item, encoder_len_count_item = encoder_acc(temp_encoder_preds,
                                                                                           temp_encoder_labels)
            encoder_len_count += 1
            encoder_len_match += encoder_len_item
            encoder_preds_count += encoder_len_match_item
            encoder_label_count += encoder_len_count_item

            if temp_label in prediction_list:
                labels_index.append(prediction_list.index(temp_label))
            else:
                labels_index.append(len(prediction_list))

            if temp_pred in prediction_list:
                preds_index.append(prediction_list.index(temp_pred))
            else:
                preds_index.append(len(prediction_list))

        micro_accuracy = accuracy_score(labels_index, preds_index)
        conf_mat = confusion_matrix(labels_index, preds_index, labels=[0, 1, 2, 3])
        conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        macro_accuracy = np.mean([conf_mat_norm[i][i] for i in range(conf_mat.shape[0])])
        print("----metrics----")
        print(conf_mat)
        print(conf_mat_norm)
        return micro_accuracy, macro_accuracy, encoder_len_match / max(encoder_len_count, 1), encoder_preds_count / max(
            1, encoder_label_count), preds_index

        # return (yes_pred + no_pred + irr_pred + ac_pred) / max((yes_count + no_count + irr_count + ac_count), 1), \
        #        yes_pred / max(yes_count, 1), \
        #        no_pred / max(no_count, 1), \
        #        irr_pred / max(irr_count, 1), \
        #        ac_pred / max(ac_count, 1), \
        #        encoder_preds_count / max(encoder_label_count, 1), \
        #        encoder_len_match / max(encoder_len_count, 1)

    def split_not_none(preds, labels):
        bleu_preds = []
        bleu_labels = []
        for i in range(len(labels)):
            # not in ['irrelevant', 'yes', 'no']
            temp_label = labels[i].split(split_str)[-1]
            temp_pred = preds[i].split(split_str)[-1]
            # temp_label = labels[i]
            if temp_label.strip().lower() not in prediction_list:
                bleu_labels.append(temp_label.strip().lower())
                bleu_preds.append(preds[i].split(split_str)[-1].strip().lower())

        return bleu_preds, bleu_labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        len_preds = len(preds)
        if len_preds == 3:
            preds, entailment_preds, encoder_preds = preds[0], torch.tensor(preds[1]), preds[2]
            labels, entailment_labels, encoder_labels = labels[0], torch.tensor(labels[1]), labels[2]

            entailment_preds = entailment_preds.masked_select(entailment_preds != -100).numpy()
            entailment_labels = entailment_labels.masked_select(entailment_labels != -100).numpy()

            encoder_entailment_acc = accuracy_score(entailment_labels, entailment_preds)
            encoder_classify_acc = accuracy_score(encoder_labels, encoder_preds)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        decoded_labels = [item.split(split_str)[-1].strip() for item in decoded_labels]
        decoded_preds = [item.split(split_str)[-1].strip() for item in decoded_preds]
        print(" labels: ", decoded_labels[:10])
        print(" decode: ", decoded_preds[:10])

        # bleu_preds, bleu_labels = split_not_none(preds=decoded_preds, labels=decoded_labels)
        #
        # print(" bleu_label: ", bleu_preds[:10])
        # print(" bleu_decode: ", bleu_labels[:10])

        result = metric.evaluate(y_true=decoded_labels, y_pred=decoded_preds)
        # result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        # _, _, result["edu_len_acc"], result[
        #     "decoder_entailment_acc"], preds_index = acc_metric(
        #     preds=decoded_preds,
        #     labels=decoded_labels)

        # result["micro_accuracy"], result["macro_accuracy"], result["edu_len_acc"], result[
        #     "decoder_entailment_acc"], preds_index = acc_metric(
        #     preds=decoded_preds,
        #     labels=decoded_labels)

        if len_preds == 3:
            result["encoder_entailment_acc"] = encoder_entailment_acc
            result["encoder_classify_acc"] = encoder_classify_acc
            # result["encoder_decoder_acc"] = accuracy_score(encoder_preds, preds_index)

        # result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = Seq2SeqEncoderParallelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "dev_generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

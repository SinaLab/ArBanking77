# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning a ?? Transformers model for sequence classification on GLUE."""
import os
import random
import logging
import argparse

import torch
import datasets
import transformers

from tqdm import tqdm
from accelerate import Accelerator
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          default_data_collator, set_seed)

from utils import write_classification_report, write_confusion_matrix, calculate_metrics, print_metrics, \
    save_dict_as_csv_pandas, save_any_dict_into_json


def add_fh_to_logger(logger, log_path, logging_format):
    if log_path == "" or log_path is None:
        return logger
    # logger.addHandler(logging.StreamHandler())
    formatter = logging.Formatter(logging_format)
    log_file_path = os.path.join(log_path, "log.txt")
    fh = logging.FileHandler(log_file_path, 'wt')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to the results folder where the model evaluation results will be saved",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the ?? Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Write logs to a file."
    )
    args = parser.parse_args()

    # Sanity checks
    if args.validation_file is None:
        raise ValueError("Need either a validation file.")
    else:
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.log_path is not None:
        os.makedirs(args.log_path, exist_ok=True)

    if args.results_dir is not None:
        os.makedirs(args.results_dir, exist_ok=True)

    return args


def main():
    args = parse_args()
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logger = logging.getLogger(__name__)

    logging_format = "%(asctime)s` - %(levelname)s - %(name)s -   %(message)s"
    logging.basicConfig(
        format=logging_format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = add_fh_to_logger(logger, args.log_path, logging_format)

    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    # Loading the dataset from local csv or json file.
    data_files = {"validation": args.validation_file}
    extension = args.validation_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels

    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    is_regression = raw_datasets["validation"].features["label"].dtype in ["float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = raw_datasets["validation"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)

    # Preprocessing the datasets

    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    non_label_column_names = [name for name in raw_datasets["validation"].column_names if name not in ["label", "tag"]]
    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
        sentence1_key, sentence2_key = "sentence1", "sentence2"
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

    label_to_id = {v: i for i, v in enumerate(label_list)}
    # print(label_to_id)

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["validation"].column_names
    )

    eval_dataset = processed_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(eval_dataset)), 3):
        logger.info(f"Sample {index} of the validation set: {eval_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(
        model, eval_dataloader
    )

    # Get the metric function
    eval_metric = load_metric("accuracy")

    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []
    all_texts = []
    for batch in tqdm(eval_dataloader):
        outputs = model(**batch)
        inputs = batch["input_ids"]
        predictions = outputs.logits.argmax(dim=-1)
        softmax_outputs = torch.nn.functional.softmax(outputs.logits, dim=1)

        eval_metric.add_batch(
            predictions=accelerator.gather(predictions),
            references=accelerator.gather(batch["labels"]),
        )
        all_labels.extend(accelerator.gather(batch["labels"]).cpu().numpy().tolist())
        all_predictions.extend(accelerator.gather(predictions).cpu().numpy().tolist())
        all_probs.extend(softmax_outputs.detach().cpu().numpy().tolist())
        all_texts.extend([tokenizer.decode(inp, skip_special_tokens=True) for inp in inputs])

    # save id_to_label into json file
    id_to_label = {id: label for label, id in label_to_id.items()}
    id_to_label_path = os.path.join(args.results_dir, "id_to_label.json")
    save_any_dict_into_json(id_to_label, id_to_label_path)

    # calculate and write metrics
    metrics_dict = calculate_metrics(all_labels, all_predictions)
    print_metrics(metrics_dict)

    test_metrics_csv_path = os.path.join(args.results_dir, "test_metrics.xlsx")
    save_dict_as_csv_pandas([metrics_dict], test_metrics_csv_path, transpose=True)

    # write conf_mat, cls_report and log_results into files
    test_confusion_matrix_path = os.path.join(args.results_dir, "test_conf_matrix.txt")
    write_confusion_matrix(all_labels, all_predictions, test_confusion_matrix_path)

    test_classification_report_path = os.path.join(args.results_dir, "test_class_report.xlsx")
    write_classification_report(all_labels, all_predictions, test_classification_report_path)

    eval_acc = eval_metric.compute()['accuracy']
    logger.info(f"eval_acc: {eval_acc}\t")

    return eval_acc, metrics_dict


if __name__ == '__main__':
    main()

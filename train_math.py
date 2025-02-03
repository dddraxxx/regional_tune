#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#    Modified by Zheng Yuan and Hongyi Yuan

import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import io
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import argparse
import json
import random;random.seed(42)
import wandb  # Add wandb import
import re

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request."
        "End your response with 'The answer is: [your answer]'.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
    "End your response with 'The answer is: [your answer]'.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
#### 28
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_length: int = field(default=None, metadata={"help": "Number of samples to use"})
    data_percent: float = field(default=100.0, metadata={"help": "Percentage of data to use (0-100)"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    overwrite_output_dir: bool = field(default=True)
    tune_layers: str = field(
        default="all",
        metadata={
            "help": "Layers to tune. Format: 'all' | '3' | '3-7' | '1,3,5' | '1-3,5,7-9'"
        },
    )
    report_to: str = field(
        default="wandb",
        metadata={"help": "Report to wandb or not"},
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        data_path = data_args.data_path
        try:
            data_path = data_path_map[data_path]
        except:
            data_path = data_path
        try:
            list_data_dict = jload(data_path)
        except BaseException:
            with open(data_path, 'r') as f:
                lines = f.readlines()
            list_data_dict = [json.loads(line.strip()) for line in lines]

        # Store original data length and data percent for logging
        self.original_data = list_data_dict
        self.data_percent = data_args.data_percent

        # Apply data percentage before random sampling
        if data_args.data_percent < 100:
            num_samples = int(len(list_data_dict) * data_args.data_percent / 100)
            list_data_dict = list_data_dict[:num_samples]

        # Random sampling
        list_data_dict = random.sample(list_data_dict, len(list_data_dict))
        if data_args.data_length is not None:
            list_data_dict = list_data_dict[:data_args.data_length]

        self.list_data_dict = list_data_dict

        # logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # print(list_data_dict[0])
        def get_input(query):
            if query.find('\n') == -1:
                return ''
            return '\n'.join(query.split('\n')[1:])
        if 'instruction' in list_data_dict[0]:
            pass
        elif 'question' in list_data_dict[0]:  # GSM8K format
            # Process GSM8K answers to match our required answer format
            list_data_dict = [{
                'instruction': data['question'],  # Full problem as instruction
                'input': '',                      # No separate input needed
                'output': f"{data['answer'].rsplit('####', 1)[0].strip()} The answer is: {data['answer'].split('####')[-1].strip()}"
            } for data in list_data_dict]
        else:  # MetaMath format with 'query'
            list_data_dict = [{'instruction':data['query'].split('\n')[0],
                             'input':get_input(data['query']),
                             'output':data['response']}
                            for data in list_data_dict]

        self.list_data_dict = list_data_dict
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def parse_layer_spec(layer_spec: str, num_layers: int) -> set[int]:
    """Parse layer specification string into set of layer indices.

    Args:
        layer_spec (str): Layer specification in formats:
            - 'all': all layers
            - Single number: '3'
            - Range: '3-7'
            - Comma-separated: '1,3,5'
            - Mixed: '1-3,5,7-9'
        num_layers (int): Total number of layers in model

    Returns:
        set[int]: Set of layer indices to tune
    """
    if layer_spec.lower() == 'all':
        return set(range(num_layers))

    layers = set()
    for part in layer_spec.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))

    # Validate indices
    if max(layers) >= num_layers or min(layers) < 0:
        raise ValueError(f"Layer indices must be between 0 and {num_layers-1}")

    return layers

def freeze_layers(model: transformers.PreTrainedModel, tune_layers: str):
    """Freeze all layers except those specified in tune_layers.

    Args:
        model: The transformer model
        tune_layers: Layer specification string
    """
    # Get number of layers based on model architecture
    if hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
    else:
        raise ValueError("Could not determine number of layers in model")

    layers_to_tune = parse_layer_spec(tune_layers, num_layers)

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specified layers
    tuned_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        should_tune = False

        # Check if parameter belongs to a transformer layer
        if '.layers.' in name:
            layer_idx = int(name.split('.layers.')[1].split('.')[0])
            should_tune = layer_idx in layers_to_tune
        # Always tune layer norms and final layer
        elif any(x in name for x in ['norm', 'lm_head', 'embed_tokens']):
            should_tune = True

        if should_tune:
            param.requires_grad = True
            tuned_params += param.numel()

    logging.info(f"Tuning {len(layers_to_tune)}/{num_layers} layers: {sorted(layers_to_tune)}")
    logging.info(f"Trainable params: {tuned_params:,} ({tuned_params/total_params:.1%} of total)")
    logging.info("Note: Layer normalization and output layer parameters are always tuned")

def log_model_parameters(model: transformers.PreTrainedModel, log_file: str = "model_params.log"):
    """Log detailed information about model parameters.

    Args:
        model: The transformer model
        log_file: Path to output log file
    """
    total_params = 0
    trainable_params = 0

    # Open log file
    with open(log_file, 'w') as f:
        f.write(f"{'Parameter Name':<60} {'Shape':<20} {'Trainable':<10} {'#Params':<12}\n")
        f.write("-" * 102 + "\n")

        # Log details for each parameter
        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params

            # Format shape as string
            shape_str = str(tuple(param.shape))

            # Write parameter details
            f.write(f"{name:<60} {shape_str:<20} {str(param.requires_grad):<10} {num_params:<12,}\n")

        # Write summary
        f.write("\n" + "=" * 102 + "\n")
        f.write(f"Total Parameters:      {total_params:,}\n")
        f.write(f"Trainable Parameters:  {trainable_params:,} ({trainable_params/total_params:.2%})\n")
        f.write(f"Frozen Parameters:     {total_params-trainable_params:,} ({1-trainable_params/total_params:.2%})\n")

    logging.info(f"Parameter details logged to {log_file}")

def log_dataset_statistics(dataset, log_file: str = "dataset_stats.log"):
    """Log detailed information about dataset statistics.

    Args:
        dataset: The SupervisedDataset instance
        log_file: Path to output log file
    """
    with open(log_file, 'w') as f:
        f.write("Dataset Statistics\n")
        f.write("=" * 50 + "\n\n")

        # Basic counts
        total_samples = len(dataset.original_data)
        used_samples = len(dataset)
        f.write(f"Total samples in original dataset: {total_samples:,}\n")
        f.write(f"Samples used after {dataset.data_percent}% selection: {used_samples:,}\n\n")

        # Analyze instruction/input/output stats
        instruction_lengths = []
        input_lengths = []
        output_lengths = []
        samples_with_input = 0

        for item in dataset.list_data_dict:
            instruction_lengths.append(len(item['instruction'].split()))
            if item.get('input', '').strip():
                samples_with_input += 1
                input_lengths.append(len(item['input'].split()))
            output_lengths.append(len(item['output'].split()))

        # Write length statistics
        f.write("Length Statistics (in words):\n")
        f.write("-" * 30 + "\n")
        f.write(f"Instructions:\n")
        f.write(f"  - Average length: {sum(instruction_lengths)/len(instruction_lengths):.1f}\n")
        f.write(f"  - Min length: {min(instruction_lengths)}\n")
        f.write(f"  - Max length: {max(instruction_lengths)}\n\n")

        if input_lengths:
            f.write(f"Inputs:\n")
            f.write(f"  - Average length: {sum(input_lengths)/len(input_lengths):.1f}\n")
            f.write(f"  - Min length: {min(input_lengths)}\n")
            f.write(f"  - Max length: {max(input_lengths)}\n")
            f.write(f"  - Samples with input: {samples_with_input:,} ({samples_with_input/used_samples:.1%})\n\n")

        f.write(f"Outputs:\n")
        f.write(f"  - Average length: {sum(output_lengths)/len(output_lengths):.1f}\n")
        f.write(f"  - Min length: {min(output_lengths)}\n")
        f.write(f"  - Max length: {max(output_lengths)}\n")

    logging.info(f"Dataset statistics logged to {log_file}")

def print_dataset_examples(dataset, num_examples=1):
    """Print example data from the dataset.

    Args:
        dataset: SupervisedDataset instance
        num_examples: Number of examples to print
    """
    logging.info(f"\n{'='*40} Dataset Examples {'='*40}")
    for i in range(min(num_examples, len(dataset.list_data_dict))):
        example = dataset.list_data_dict[i]
        logging.info(f"\nExample {i+1}:")
        logging.info(f"Instruction: {example['instruction']}")
        logging.info(f"Input: {example.get('input', '')}")
        logging.info(f"Output: {example['output']}")
        logging.info(f"\nFormatted Source:")
        logging.info(dataset.sources[i])
        logging.info(f"Target: {dataset.targets[i]}")
        logging.info(f"{'-'*90}")

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # Initialize wandb
    if training_args.report_to == "wandb":
        # Extract dataset name from data path
        dataset_name = os.path.basename(data_args.data_path).split('.')[0]
        output_dir_base = training_args.output_dir.split('/')[-1],
        wandb.init(
            name=f"{dataset_name}.{output_dir_base}",
            config={
                "model": model_args.model_name_or_path,
                "data_path": data_args.data_path,
                "data_length": data_args.data_length,
                "data_percent": data_args.data_percent,
                "tune_layers": training_args.tune_layers,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "max_steps": training_args.max_steps,
            }
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Print example data
    print_dataset_examples(data_module['train_dataset'])

    # Add layer freezing after model loading
    if training_args.tune_layers != "all":
        freeze_layers(model, training_args.tune_layers)
    # Log parameter details after freezing
    os.makedirs(training_args.output_dir, exist_ok=True)
    log_model_parameters(model, os.path.join(training_args.output_dir, "model_params.log"))

    # Log model parameter stats to wandb
    if training_args.report_to == "wandb":
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.log({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": total_params - trainable_params,
            "trainable_percentage": trainable_params / total_params * 100,
        })

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if training_args.report_to == "wandb":
        wandb.finish()

    # log_dataset_statistics(data_module['train_dataset'])

if __name__ == "__main__":
    train()
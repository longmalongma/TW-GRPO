# Copyright 2025 The HuggingFace Team. All rights reserved.
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
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys

import datasets
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
import trl
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from transformers.utils import is_peft_available
if is_peft_available():
    from peft import PeftConfig, get_peft_model

from qwen_vl_utils import process_vision_info
logger = logging.getLogger(__name__)


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})


@dataclass
class ScriptArguments(trl.ScriptArguments):
    """
    Additional arguments for training
    """
    jsonl_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training data jsonl file"}
    )


processor = None

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively."
)

QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."

def make_conversation(example):
    """
    Format text-only conversation
    """
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": example["problem"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"{example.get('thinking', '')}\n<answer>{example.get('model_answer', '')}</answer>"}]},
        ],
    }

def make_conversation_image(example):
    """
    Format conversation with image
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"{example.get('thinking', '')}\n<answer>{example.get('model_answer', '')}</answer>"}],
            },
        ],
    }

def make_conversation_video(example):
    """
    Format conversation with video, matching test_qwen2vl implementation
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "text": example["video"]},  # Add video path
                    {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": f"{example.get('thinking', '')}\n<answer>{example.get('model_answer', '')}</answer>"}]
            }
        ]
    }

def collate_fn(examples):
    """
    Collate function for the dataloader
    """
    texts = []
    video_inputs = []
    
    for example in examples:
        messages = example["messages"]
        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        
        # Process video path
        msg = messages[0]  # User message
        contents = msg['content']
        video_path = None
        for content in contents:
            if content['type'] == 'video':
                video_path = content['text']
                break
                
        if video_path:
            # Create temporary message format for process_vision_info
            temp_msg = [{
                "role": "user",
                "content": [
                    {"type": "video", "text": video_path},
                    {"type": "text", "text": ""}
                ]
            }]
            video_input = process_vision_info(temp_msg)[0]
            video_inputs.append(video_input)
        else:
            video_inputs.append(None)
    
    # Create batch
    batch = processor(
        text=texts,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # Process labels
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    if hasattr(processor, "tokenizer"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch

from datasets import Dataset, DatasetDict
import json

def create_dataset_from_jsonl_simple(jsonl_path):
    base_dataset = Dataset.from_json(jsonl_path)
    return DatasetDict({
        "train": base_dataset
    })

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################
    if script_args.jsonl_path:
        dataset = create_dataset_from_jsonl_simple(script_args.jsonl_path)
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Convert examples based on data type
    if "image" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(make_conversation_image)
    elif "video" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(make_conversation_video)
    else:
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    ################
    # Load processor
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        )
        logger.info("Using AutoTokenizer for text-only model.")

    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # training_args.model_init_kwargs = model_kwargs
    from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
    if "Qwen2-VL" in model_args.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    elif "Qwen2.5-VL" in model_args.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, **model_kwargs
        )
    else:
        assert False, f"Model {model_args.model_name_or_path} not supported"
    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    peft_config = get_peft_config(model_args)
    freeze_vision_modules = True
    vision_modules_keywords = ["visual"]
    if peft_config is not None:
        def find_all_linear_names(model, multimodal_keywords):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                 # LoRA is not applied to the vision modules
                if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                    continue
                if isinstance(module, cls):
                    lora_module_names.add(name)
            for m in lora_module_names:  # needed for 16-bit
                if "embed_tokens" in m:
                    lora_module_names.remove(m)
            return list(lora_module_names)
        target_modules = find_all_linear_names(model, vision_modules_keywords)
        peft_config.target_modules = target_modules
        model = get_peft_model(model, peft_config)

    if freeze_vision_modules:
        print("Freezing vision modules...")
        for n, p in model.named_parameters():
            if any(keyword in n for keyword in vision_modules_keywords):
                p.requires_grad = False

    training_args.remove_unused_columns = False
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=peft_config
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["R1-V"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    #############
    # push to hub
    #############

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
        processor.push_to_hub(training_args.hub_model_id)




if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
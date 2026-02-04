from typing import List
import torch
from transformers import AutoProcessor
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Optional
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
import peft
import transformers
from PIL import Image

from clarification_trees.dialog_tree import DialogTrajectory

class TransformersModel:
    """

    """
    adapted_model: peft.PeftModel | peft.PeftMixedModel | None = None  # Model including any LORAs
    base_model: transformers.PreTrainedModel  # Base model without any LORAs
    bnb_config: Optional[BitsAndBytesConfig]
    max_new_tokens: int

    def __init__(self, model_config: DictConfig, device: str):
        self.model_config = model_config
        self.model_name = model_config.model_name
        self.device = device
        self.max_new_tokens = model_config.max_new_tokens

        self.image_resize_config = model_config.get("image_resize_config", None)
        if self.image_resize_config:
            print(f"Image resizing enabled: {self.image_resize_config}")

        if "bnb_config" in model_config:
            print("Loading model with BNB config")
            self.bnb_config = BitsAndBytesConfig(**model_config.bnb_config)
        else:
            self.bnb_config = None

        self.base_model, self.processor = self._load_base_model(self.model_config, self.bnb_config)

    ### BASE MODEL MANAGEMENT ###
    def _load_base_model(self, model_config: DictConfig, bnb_config: Optional[BitsAndBytesConfig] = None) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        if model_config.model_name == "qwen-3-vl-2b":
            base_model, processor = self._load_qwen_vl_model(model_config, bnb_config)
        elif model_config.model_name == "qwen-3-vl-4b":
            base_model, processor = self._load_qwen_vl_model(model_config, bnb_config)
        elif model_config.model_name == "qwen-3-vl-8b":
            base_model, processor = self._load_qwen_vl_model(model_config, bnb_config)
        elif model_config.model_name == "qwen-3-vl-32b":
            base_model, processor = self._load_qwen_vl_model(model_config, bnb_config)
        elif model_config.model_name == "qwen-3-vl-235b":
            base_model, processor = self._load_qwen_vl_model(model_config, bnb_config)
        else:
            raise NotImplementedError(f"Model {model_config.model_name} is not implemented")

        return base_model, processor

    def _load_qwen_vl_model(self, model_config: DictConfig, bnb_config: Optional[BitsAndBytesConfig] = None) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError:
            raise ImportError("Qwen3VLForConditionalGeneration is not available. Please install transformers.")
        
        if model_config.use_flash_attention:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_config.model_hf_transformers_key,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=self.device,
                quantization_config=self.bnb_config
            )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_config.model_hf_transformers_key,
                dtype="auto",
                device_map=self.device,
                quantization_config=self.bnb_config
            )
        processor = AutoProcessor.from_pretrained(model_config.model_hf_transformers_key)

        return model, processor

    ### LORA ADAPTER MANAGEMENT ###
    def load_adapter(self, adapter_load_dir: Path) -> peft.PeftModel | peft.PeftMixedModel:
        """
        Loads a LORA adapter from a path and applies it to the base model.
        """
        print(f"Loading adapter from {adapter_load_dir}")
        self.adapted_model = peft.PeftModel.from_pretrained(
            self.base_model,
            adapter_load_dir.absolute().as_posix(),
            is_trainable=True
        )
        return self.adapted_model

    def save_adapter(self, adapter_save_dir: Path) -> None:
        """
        Saves the LORA adapter to a path.
        """
        if self.adapted_model is None:
            raise ValueError("Cannot save adapter: No adapter is currently loaded or constructed.")
        
        print(f"Saving adapter to {adapter_save_dir}")
        self.adapted_model.save_pretrained(adapter_save_dir.absolute().as_posix())

    def construct_new_adapter(self, lora_config: DictConfig) -> peft.PeftModel | peft.PeftMixedModel:
        """
        Constructs a new LORA adapter from a config.
        """
        lora_model = prepare_model_for_kbit_training(self.base_model)
        config_obj = LoraConfig(**OmegaConf.to_container(lora_config))
        print(f"Adding LoRA to model with config: {config_obj}")
        self.adapted_model = get_peft_model(lora_model, config_obj)
        return self.adapted_model

    ### GENERATION & DIALOG TREE ###
    def _pad_and_resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resizes an image to fit within target dimensions while maintaining aspect ratio,
        then pads the remaining space to ensure exact output dimensions.
        """
        if not self.image_resize_config:
            return image

        target_w = self.image_resize_config.width
        target_h = self.image_resize_config.height
        
        # Default to black padding if not specified
        pad_color = tuple(self.image_resize_config.get("pad_color", [0, 0, 0]))

        original_w, original_h = image.size
        ratio = min(target_w / original_w, target_h / original_h)
        new_w = int(original_w * ratio)
        new_h = int(original_h * ratio)

        # Resize with high-quality downsampling
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Create new background image
        new_image = Image.new("RGB", (target_w, target_h), pad_color)
        
        # Paste resized image in the center
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        new_image.paste(image, (paste_x, paste_y))

        return new_image

    def _process_images_in_messages(self, messages: list[dict]):
        """
        Iterates over the message structure (list of dicts) used by `apply_chat_template`.
        Finds PIL Images and replaces them with padded/resized versions.
        """
        if not self.image_resize_config:
            return messages

        for message in messages:
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image":
                        # Some templates use "image" key with PIL object, some use "image_url"
                        # Adjust based on how DialogTrajectory stores it. 
                        # Assuming 'image' key holds the PIL object based on standard VLM usage.
                        if "image" in item and isinstance(item["image"], Image.Image):
                            item["image"] = self._pad_and_resize_image(item["image"])
        return messages

    def preprocess_generation_inputs(self, trajectory: DialogTrajectory, base_prompt_override: str | None = None, as_user: bool = False):
        messages = trajectory.to_messages(model_name=self.model_name, reverse_roles=False)
        messages.insert(0, {"role": "system", "content": [{"type": "text", "text": base_prompt_override or self.model_config.base_prompt}]})
        messages = self._process_images_in_messages(messages)
        if not as_user:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
        else:
            # # Then we add a dummy message to the end that is empty and use the continue generation option
            # messages.append({"role": "user", "content": [{"type": "text", "text": "Regarding your question, I am asking about "}]})
            # inputs = self.processor.apply_chat_template(
            #     messages,
            #     tokenize=True,
            #     add_generation_prompt=False,
            #     continue_final_message=True,
            #     return_dict=True,
            #     return_tensors="pt"
            # )
            # Add system prompt specific to the User Persona
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": base_prompt_override}]})
            messages = self._process_images_in_messages(messages)
            
            # Instead of appending a User message and asking to complete it, 
            # we append a User message creating a "Simulation Task" for the Assistant.
            
            # Current History: [User (Ambiguous), Assistant (Clarifying)]
            
            # We add a clear instruction for the model to generate the response
            prompt_text = "Please write the response to the clarifying question above as if you were the user described in the system prompt. Reply directly with the clarification text only. Your text should start with something like 'Regarding your question, I am asking about ', 'Yes, ...' or 'No, ...'. Be short and concise."
            messages.append({"role": "user", "content": [{"type": "text", "text": prompt_text}]})
            
            # Now the model generates an 'Assistant' turn, which contains the 'User's' text.
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True, # Standard generation
                return_dict=True,
                return_tensors="pt"
            )
        print(f"Generating using messages: {messages}")
        inputs = inputs.to(self.device)
        return inputs

    def preprocess_training_inputs(self, trajectory: DialogTrajectory, base_prompt_override: str | None = None, reverse_roles: bool = False):
        """
        Prepares inputs and labels for Causal LM training.
        The final message is the target while every message before it is context and has the label masked.
        """
        messages = trajectory.to_messages(model_name=self.model_name, reverse_roles=reverse_roles)  # Get the list of dict context formatted correctly

        # Add system prompt
        messages.insert(0, {"role": "system", "content": [{"type": "text", "text": base_prompt_override or self.model_config.base_prompt}]})

        # If we need to, resize and pad images
        messages = self._process_images_in_messages(messages)

        # To get which tokens are context and which are learnable, we tokenize twice, once with the generation prompt and only the previous context
        # and once with the full context without a new generation prompt.
        # By then masking out the length of the one with the generation prompt, we can get a mask on the labels that only has the learnable tokens unmasked.
        full_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False, # We want the full text, not a prompt for generation
            return_dict=True,
            return_tensors="pt"
        )

        prompt_messages = messages[:-1] # Everything except the final assistant response
        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True, # Add the token that triggers assistant generation
            return_dict=True,
            return_tensors="pt"
        )

        input_ids = full_inputs["input_ids"][0]
        labels = input_ids.clone()

        # Mask out the prompt part in the labels
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_len] = -100  # -100 is the ignore_index for CrossEntropyLoss

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": full_inputs["attention_mask"][0],
            "pixel_values": full_inputs["pixel_values"],
            "image_grid_thw": full_inputs["image_grid_thw"][0]
        }

        # id_to_token_map = {v: k for k, v in self.processor.tokenizer.get_vocab().items()}
        # id_to_token_map[-100] = "<mask>"
        # input_ids_to_string = [id_to_token_map.get(int(id), "") for id in input_ids]
        # labels_to_string = [id_to_token_map.get(int(id), "") for id in labels]

        return result

    def generate(self, trajectory: DialogTrajectory, base_prompt_override: Optional[str] = None, use_base_model: bool = False, as_user: bool = False):
        inputs = self.preprocess_generation_inputs(trajectory, base_prompt_override, as_user)
        if use_base_model or not self.model_config.lora_config.use_lora:
            model = self.base_model
        else:
            assert self.adapted_model is not None, "Model does not have adapter loaded"
            model = self.adapted_model
        generated_ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        return generated_text
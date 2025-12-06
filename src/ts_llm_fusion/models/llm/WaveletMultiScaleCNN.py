#
#

from einops import rearrange
from types import SimpleNamespace
from src.ts_llm_fusion.models.encoder.CNNTokenizer import CNNTokenizer
#from src.ts_llm_fusion.models.encoder.WaveletTransformPS import WaveletMultiScaleEncoder
from src.ts_llm_fusion.models.encoder.MultiScaleTransformer import TSFormer

from src.ts_llm_fusion.models.llm.TimeSeriesFlamingoWithTrainableEncoder import (
    TimeSeriesFlamingoWithTrainableEncoder,
)
from src.ts_llm_fusion.models.open_flamingo.open_flamingo.src.flamingo_lm import FlamingoLMMixin
from src.ts_llm_fusion.models.open_flamingo.open_flamingo.src.utils import extend_instance
from src.ts_llm_fusion.models.encoder.MultiScaleTransformer import TSFormer
from src.ts_llm_fusion.models.encoder.MultiScaleTransformer import ConvLinear
from src.ts_llm_fusion.models.encoder.ResNetBlock import Res2Block1D, Res2Net1D
import torch
import torch.nn as nn
import torch._dynamo
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.ts_llm_fusion.utils.model_config import ENCODER_OUTPUT_DIM
from src.ts_llm_fusion.models.llm.TimeSeriesLLM import TimeSeriesLLM
from src.ts_llm_fusion.utils.prompt.full_prompt import FullPrompt
from src.ts_llm_fusion.data.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

# Monkey-patch FlamingoLayer to add attention_type property for compatibility with newer transformers
from src.ts_llm_fusion.models.open_flamingo.open_flamingo.src.flamingo_lm import FlamingoLayer


def _attention_type_property(self):
    """Proxy the attention_type attribute from the underlying decoder layer."""
    return getattr(self.decoder_layer, "attention_type", None)


# Add the attention_type property to FlamingoLayer
FlamingoLayer.attention_type = property(_attention_type_property)

class WaveletMultiScaleCNN(TimeSeriesLLM):
    def __init__(
            self,
            num_features: int,
            len_sequence: int,
            hidden_dim1: int,
            hidden_dim2: int,
            num_llm_layers: int,
            patch_size: int = 4,
            device: str = 'cuda',
            llm_id: str = "meta-llama/Llama-3.2-1B",
            input_enc: int = None):
        super().__init__(device)
        self.is_first = True

        self.initial_conv = nn.Conv1d(in_channels = num_features, out_channels=1, kernel_size=4, stride=4)

        text_tokenizer = AutoTokenizer.from_pretrained(
            llm_id,
            local_files_only=False,
            trust_remote_code=True,
            cache_dir=None,
        )
        

        lang_encoder = AutoModelForCausalLM.from_pretrained(
            llm_id,
            local_files_only=False,
            trust_remote_code=True,
            cache_dir=None,
            device_map={"": device},
            attn_implementation="eager",
        )
        extend_instance(lang_encoder, FlamingoLMMixin)
        lang_encoder.resize_token_embeddings(len(text_tokenizer))

        time_model = AutoModelForCausalLM.from_pretrained(
            llm_id,
            local_files_only=False,
            trust_remote_code=True,
            cache_dir=None,
            device_map={"": device},
            attn_implementation="eager",
        )
        self.time_model = time_model
        self.text_model = time_model
        self.llm = lang_encoder


        self.time_proj = nn.ModuleList(
            [nn.Linear(hidden_dim2, hidden_dim2, bias=False) for _ in range(num_llm_layers + 1)
        ])
        self.text_proj = nn.ModuleList(
            [nn.Linear(hidden_dim2, hidden_dim2, bias=False) for _ in range(num_llm_layers + 1)
        ])
        # word_embedding = torch.tensor(torch.load(configs.word_embedding_path)).to(device=device)
        self.in_layer = ConvLinear(len_sequence, hidden_dim1, 1)
        self.out_layer = nn.Linear(hidden_dim2 * hidden_dim1, lang_encoder.config.hidden_size)

        self.time_series_encoder = TSFormer(self, patch_size=patch_size, in_channels=1, embed_dim=hidden_dim1,
                                       num_heads=4, mlp_ratio=4, dropout=0.1, num_token=4032 / patch_size,
                                       mask_ratio=-0.75, encoder_depth=4, decoder_depth=1, stage='forecasting',
                                       embed_path='your_model_path/clean_word_embeddings.pt')
        #time_series_encoder_path = configs.ts_encoder_path
        #word_embedding = torch.tensor(torch.load('your_model_path/clean_word_embeddings.pt')).to(device=device)
        #related_words = torch.load('your_model_path/clean_related_words.pt')

        self.basic_conv = Res2Net1D(hidden_dim1, 64, 2, 16)
        self.basic_conv.to(device=device)
        self.basic_conv.train()

        #time_series_encoder.load_state_dict(torch.load(time_series_encoder_path)['model_state_dict'])
        self.time_series_encoder.to(device=device)
        self.time_series_encoder.eval()
        self.in_layer.to(device=device)
        self.in_layer.train()

    def pad_and_apply_batch(
        self, batch: List[Dict[str, any]], include_labels: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def pad_time_series(batch, max_length=None):
            """Pad time series to the same length (either max in batch or specified max)"""
            time_series = [item["time_series"] for item in batch]

            # Determine target length (either specified or max in batch)
            if max_length is None:
                max_length = max(ts.shape[1] for ts in time_series)

            padded_series = []
            for ts in time_series:
                current_length = ts.shape[1]
                if current_length < max_length:
                    # Pad with zeros to reach max_length
                    # Ensure padding has the same number of dimensions as the time series
                    padding_shape = list(ts.shape)
                    padding_shape[1] = max_length - current_length
                    padding = torch.zeros(
                        padding_shape, device=ts.device, dtype=ts.dtype
                    )
                    padded = torch.cat([ts, padding], dim=1)
                else:
                    # If already at or exceeding max_length, truncate
                    padded = ts[:, :max_length]

                padded_series.append(padded)

            return torch.stack(padded_series)

        cast_dtype = None
        tokenizer = self.text_tokenizer
        media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
            "input_ids"
        ][-1]

        # Process time series data
        images = pad_time_series(batch).to(
            self.device, dtype=cast_dtype, non_blocking=True
        )
        images = images.unsqueeze(1)  # Add time dimension


        # Process text inputs WITH answers
        text_inputs = []
        prompt_lengths = []

        for item in batch:
            # Build the prompt text without answer
            prompt_text = item["pre_prompt"]
            for ts_text in item["time_series_text"]:
                prompt_text += f" {tokenizer.decode([media_token_id])} {ts_text} {tokenizer.decode([endofchunk_token_id])}"
            if item["post_prompt"]:
                prompt_text += f" {item['post_prompt']}"

            if include_labels:
                text_inputs.append(prompt_text)
                continue

            # Store the prompt length in tokens
            prompt_tokens = tokenizer(prompt_text, add_special_tokens=False).input_ids
            prompt_lengths.append(len(prompt_tokens))

            # Add the answer to create full text
            full_text = prompt_text + f" {item['answer']}"
            text_inputs.append(full_text)

        # Tokenize full text (prompt + answer)
        tokenized = tokenizer(text_inputs, padding="longest", return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.device, non_blocking=True)
        attention_mask = tokenized.attention_mask.to(self.device, non_blocking=True)

        if include_labels:
            return input_ids, images, attention_mask, None

        # Create labels matrix (-100 for masked tokens)
        labels = torch.full_like(input_ids, -100)

        # Set labels for answer tokens using the stored prompt lengths
        for i, prompt_length in enumerate(prompt_lengths):
            non_padding_indices = torch.where(input_ids[i] != tokenizer.pad_token_id)[0]
            answer_indices = non_padding_indices[non_padding_indices >= prompt_length]

            if len(answer_indices) > 0:
                labels[i, answer_indices] = input_ids[i, answer_indices]

        return input_ids, images, attention_mask, labels

    def compute_loss(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """
        batch: same format as generate()
        answers: List[str] of length B
        """
        input_ids, images, attention_mask, labels = self.pad_and_apply_batch(
            batch, include_labels=False
        )

        output = self.forecast(
            vision_x=images,
            lang_x=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return output[0]

    def forecast(self, batch):

        x, input_ids, attention_mask, labels = self.pad_and_apply_batch(batch)
        B, L, M = x.shape
        #x = self.initial_conv(x)
        x = self.rev_in(x, 'norm')
        x = rearrange(x, 'b l m -> b m l')

        outputs_time1 = self.in_layer(x)
        if self.is_first: print('Orig--'); self.is_first = False
        BS, NS, LS = x.shape
        x = x.reshape(BS, NS, 1, LS)
        outputs_text1 = self.time_series_encoder(x)

        outputs_text1 = outputs_text1.reshape(outputs_text1.shape[0], outputs_text1.shape[1],
                                              outputs_text1.shape[2]*outputs_text1.shape[3])
        outputs_time1 = self.initial_conv(outputs_time1)
        outputs_time, intermediate_feat_time = self.time_model(inputs_embeds=outputs_time1)
        outputs_text, intermediate_feat_text = self.text_model(inputs_embeds=outputs_text1)


        # Residual Connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1

        outputs_time = self.out_layer(outputs_time[:, -M, :])
        outputs_text = self.out_layer(outputs_text[:, -M, :])

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        ts_data = self.initial_conv(outputs_time)
        text_data = self.initial_conv(outputs_text)
        output = self.llm(ts_data, text_data, attention_mask, labels)

        outputs_time = self.rev_in(outputs_time, 'denorm')
        outputs_text = self.rev_in(outputs_text, 'denorm')
        outputs_time = (outputs_time + outputs_text) / 2
        return outputs_text, outputs_time, intermediate_feat_time, intermediate_feat_text, output[0]

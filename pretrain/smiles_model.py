import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForMaskedLM,RobertaModel,RobertaPreTrainedModel
from typing import List, Optional, Tuple, Union


class Smiles_encoder_model(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
    def forward(self,input_ids):
        outputs = self.roberta(
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
        )
        sequence_output = outputs[0]
        return sequence_output


class encoder_model(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.encoder = self.roberta.encoder
    def forward(self,hidden_states):
        new_embedding = self.encoder(
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        total_embedding = new_embedding[0]
        return total_embedding

class smiles_decoder_model(RobertaPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.robertamlm = RobertaForMaskedLM(config)
        self.decoder = self.robertamlm.lm_head
    def forward(self,smiles_embedding):
        prediction_scores = self.decoder(smiles_embedding)
        return prediction_scores





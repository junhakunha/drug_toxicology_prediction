import os
import sys

sys.path.append(os.getcwd())
sys.path.append('../')

from torch import nn
from transformers import RobertaForSequenceClassification


class RobertaWithEmbeddings(RobertaForSequenceClassification):
    """
    A subclass of RobertaForSequenceClassification that adds functionality
    to return embeddings from the penultimate layer while keeping all other
    behavior identical.
    """

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        return_embeddings=False,  # New flag for returning embeddings
    ):
        # Pass inputs through the Roberta model
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]  # Shape: [batch_size, seq_len, hidden_dim]

        # Extract CLS token embedding (penultimate layer)
        cls_embeddings = sequence_output[:, 0, :]  # Shape: [batch_size, hidden_dim]

        if return_embeddings:
            return cls_embeddings  # Return embeddings directly

        # Pass CLS embedding through classifier head
        logits = self.classifier(sequence_output)  # Shape: [batch_size, num_labels]

        outputs = (logits,) + outputs[2:]  # Add hidden_states and attentions if available

        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # Classification task
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
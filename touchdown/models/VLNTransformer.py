from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from texar.torch.core import layers
from texar.torch.modules.embedders.embedders import WordEmbedder
from texar.torch.modules.embedders.position_embedders import PositionEmbedder
from texar.torch.modules.encoders.encoder_base import EncoderBase
from texar.torch.modules.encoders.transformer_encoder import TransformerEncoder
from texar.torch.modules.pretrained.bert import PretrainedBERTMixin
from texar.torch.modules.classifiers.classifier_base import ClassifierBase
from texar.torch.utils.utils import dict_fetch
from texar.torch.hyperparams import HParams
from texar.torch.core.layers import get_initializer


class VLNTransformerEncoder(EncoderBase, PretrainedBERTMixin):
    r"""
    This module basically stacks
    :class:`~texar.torch.modules.WordEmbedder`,
    :class:`~texar.torch.modules.PositionEmbedder`,
    :class:`~texar.torch.modules.TransformerEncoder` and a dense
    pooler.
    """

    def __init__(self, hparams=None):
        super().__init__(hparams=hparams)

        # Segment embedding for each type of tokens
        self.segment_embedder = None
        if self._hparams.get('type_vocab_size', 0) > 0:
            self.segment_embedder = WordEmbedder(
                vocab_size=self._hparams.type_vocab_size,
                hparams=self._hparams.segment_embed)

        # Position embedding
        self.position_embedder = PositionEmbedder(
            position_size=self._hparams.position_size,
            hparams=self._hparams.position_embed)

        # The BERT encoder (a TransformerEncoder)
        self.encoder = TransformerEncoder(hparams=self._hparams.encoder)

        self.pooler = nn.Sequential(
            nn.Linear(self._hparams.encoder.dim, self._hparams.hidden_size),
            nn.Tanh())

    def reset_parameters(self):
        initialize = layers.get_initializer(self._hparams.initializer)
        if initialize is not None:
            # Do not re-initialize LayerNorm modules.
            for name, param in self.named_parameters():
                if name.split('.')[-1] == 'weight' and 'layer_norm' not in name:
                    initialize(param)

    @staticmethod
    def default_hparams():
        return {
            'pretrained_model_name': 'bert-base-uncased',
            'embed': {
                'dim': 256,
                'name': 'word_embeddings'
            },
            'vocab_size': 30522,
            'segment_embed': {
                'dim': 256,
                'name': 'token_type_embeddings'
            },
            'type_vocab_size': 2,
            'position_embed': {
                'dim': 256,
                'name': 'position_embeddings'
            },
            'position_size': 256,
            'encoder': {
                'dim': 256,
                'embedding_dropout': 0.1,
                'multihead_attention': {
                    'dropout_rate': 0.1,
                    'name': 'self',
                    'num_heads': 8,
                    'num_units': 256,
                    'output_dim': 256,
                    'use_bias': True
                },
                'name': 'encoder',
                'num_blocks': 8,
                'poswise_feedforward': {
                    'layers': [
                        {
                            'kwargs': {
                                'in_features': 256,
                                'out_features': 1024,
                                'bias': True
                            },
                            'type': 'Linear'
                        },
                        {"type": "BertGELU"},
                        {
                            'kwargs': {
                                'in_features': 1024,
                                'out_features': 256,
                                'bias': True
                            },
                            'type': 'Linear'
                        }
                    ]
                },
                'residual_dropout': 0.1,
                'use_bert_config': True
            },
            'hidden_size': 256,
            'initializer': None,
            'name': 'bert_encoder',
            '@no_typecheck': ['pretrained_model_name']
        }

    def forward(self,  # type: ignore
                inputs: Union[torch.Tensor, torch.LongTensor],
                sequence_length: Optional[torch.LongTensor] = None,
                segment_ids: Optional[torch.LongTensor] = None):
        r"""Encodes the inputs.

        Args:
            inputs: Either a **2D Tensor** of shape `[batch_size, max_time]`,
                containing the ids of tokens in input sequences, or
                a **3D Tensor** of shape `[batch_size, max_time, vocab_size]`,
                containing soft token ids (i.e., weights or probabilities)
                used to mix the embedding vectors.
            segment_ids (optional): A 2D Tensor of shape
                `[batch_size, max_time]`, containing the segment ids
                of tokens in input sequences. If `None` (default), a
                tensor with all elements set to zero is used.
            sequence_length (optional): A 1D Tensor of shape `[batch_size]`.
                Input tokens beyond respective sequence lengths are masked
                out automatically.

        Returns:
            A pair :attr:`(outputs, pooled_output)`

            - :attr:`outputs`:  A Tensor of shape
              `[batch_size, max_time, dim]` containing the encoded vectors.

            - :attr:`pooled_output`: A Tensor of size
              `[batch_size, hidden_size]` which is the output of a pooler
              pre-trained on top of the hidden state associated to the first
              character of the input (`CLS`), see BERT's paper.
        """
        word_embeds = inputs

        batch_size = inputs.size(0)
        pos_length = inputs.new_full((batch_size,), inputs.size(1),
                                     dtype=torch.int64)
        pos_embeds = self.position_embedder(sequence_length=pos_length)
        if self.segment_embedder is not None:
            if segment_ids is None:
                segment_ids = torch.zeros((inputs.size(0), inputs.size(1)),
                                          dtype=torch.long,
                                          device=inputs.device)
            segment_embeds = self.segment_embedder(segment_ids)
            inputs_embeds = word_embeds + segment_embeds + pos_embeds
        else:
            inputs_embeds = word_embeds + pos_embeds

        if sequence_length is None:
            sequence_length = inputs.new_full((batch_size,), inputs.size(1),
                                              dtype=torch.int64)

        output = self.encoder(inputs_embeds, sequence_length)

        # taking the hidden state corresponding to the first token.
        first_token_tensor = output[:, 0, :]
        pooled_output = self.pooler(first_token_tensor)

        return output, pooled_output

    @property
    def output_size(self):
        r"""The feature size of :meth:`forward` output
        :attr:`pooled_output`.
        """
        return self._hparams.hidden_size


class VLNTransformer(ClassifierBase, PretrainedBERTMixin):
    r"""Classifier based on BERT modules. Please see
    :class:`~texar.torch.modules.PretrainedBERTMixin` for a brief description
    of BERT.

    This is a combination of the
    :class:`~texar.torch.modules.BERTEncoder` with a classification
    layer. Both step-wise classification and sequence-level classification
    are supported, specified in :attr:`hparams`.
    """
    _ENCODER_CLASS = VLNTransformerEncoder

    def __init__(self, hparams=None):

        super().__init__(hparams=hparams)

        # Create the underlying encoder
        encoder_hparams = dict_fetch(hparams, self._ENCODER_CLASS.default_hparams())

        self._encoder = self._ENCODER_CLASS(hparams=encoder_hparams)

        # Create a dropout layer
        self._dropout_layer = nn.Dropout(self._hparams.dropout)

        # Create an additional classification layer if needed
        self.num_classes = self._hparams.num_classes
        if self.num_classes <= 0:
            self._logits_layer = None
        else:
            logit_kwargs = self._hparams.logit_layer_kwargs
            if logit_kwargs is None:
                logit_kwargs = {}
            elif not isinstance(logit_kwargs, HParams):
                raise ValueError("hparams['logit_layer_kwargs'] "
                                 "must be a dict.")
            else:
                logit_kwargs = logit_kwargs.todict()

            if self._hparams.clas_strategy == 'all_time':
                self._logits_layer = nn.Linear(
                    self._encoder.output_size *
                    self._hparams.max_seq_length,
                    self.num_classes,
                    **logit_kwargs)
            else:
                self._logits_layer = nn.Linear(
                    self._encoder.output_size, self.num_classes,
                    **logit_kwargs)

        if self._hparams.initializer:
            initialize = get_initializer(self._hparams.initializer)
            assert initialize is not None
            if self._logits_layer:
                initialize(self._logits_layer.weight)
                if self._logits_layer.bias:
                    initialize(self._logits_layer.bias)

        self.is_binary = (self.num_classes == 1) or \
                         (self.num_classes <= 0 and
                          self._hparams.encoder.dim == 1)

    @staticmethod
    def default_hparams():
        hparams = VLNTransformerEncoder.default_hparams()
        hparams.update({
            "num_classes": 4,
            "logit_layer_kwargs": None,
            "clas_strategy": "all_time",
            "max_seq_length": 50,  # should be the same with opts.max_t_v_len in main.py
            "dropout": 0.1,
            "name": "bert_classifier"
        })
        return hparams

    def forward(self,  # type: ignore
                inputs: Union[torch.Tensor, torch.LongTensor],
                sequence_length: Optional[torch.LongTensor] = None,
                segment_ids: Optional[torch.LongTensor] = None) \
            -> Tuple[torch.Tensor, torch.LongTensor]:
        r"""Feeds the inputs through the network and makes classification.

        The arguments are the same as in
        :class:`~texar.torch.modules.BERTEncoder`.

        Args:
            inputs: Either a **2D Tensor** of shape `[batch_size, max_time]`,
                containing the ids of tokens in input sequences, or
                a **3D Tensor** of shape `[batch_size, max_time, vocab_size]`,
                containing soft token ids (i.e., weights or probabilities)
                used to mix the embedding vectors.
            sequence_length (optional): A 1D Tensor of shape `[batch_size]`.
                Input tokens beyond respective sequence lengths are masked
                out automatically.
            segment_ids (optional): A 2D Tensor of shape
                `[batch_size, max_time]`, containing the segment ids
                of tokens in input sequences. If `None` (default), a tensor
                with all elements set to zero is used.

        Returns:
            A tuple `(logits, preds)`, containing the logits over classes and
            the predictions, respectively.

            - If ``clas_strategy`` is ``cls_time`` or ``all_time``:

                - If ``num_classes`` == 1, ``logits`` and ``pred`` are both of
                  shape ``[batch_size]``.
                - If ``num_classes`` > 1, ``logits`` is of shape
                  ``[batch_size, num_classes]`` and ``pred`` is of shape
                  ``[batch_size]``.

            - If ``clas_strategy`` is ``time_wise``:

                - ``num_classes`` == 1, ``logits`` and ``pred`` are both of
                  shape ``[batch_size, max_time]``.
                - If ``num_classes`` > 1, ``logits`` is of shape
                  ``[batch_size, max_time, num_classes]`` and ``pred`` is of
                  shape ``[batch_size, max_time]``.
        """
        enc_outputs, pooled_output = self._encoder(inputs,
                                                   sequence_length,
                                                   segment_ids)
        # Compute logits
        strategy = self._hparams.clas_strategy
        if strategy == 'time_wise':
            logits = enc_outputs
        elif strategy == 'cls_time':
            logits = pooled_output
        elif strategy == 'all_time':
            # Pad `enc_outputs` to have max_seq_length before flatten
            length_diff = self._hparams.max_seq_length - inputs.shape[1]
            logit_input = F.pad(enc_outputs, [0, 0, 0, length_diff, 0, 0])
            logit_input_dim = (self._encoder.output_size *
                               self._hparams.max_seq_length)
            logits = logit_input.view(-1, logit_input_dim)
        else:
            raise ValueError('Unknown classification strategy: {}'.format(
                strategy))

        if self._logits_layer is not None:
            logits = self._dropout_layer(logits)
            logits = self._logits_layer(logits)

        # Compute predictions
        if strategy == "time_wise":
            if self.is_binary:
                logits = torch.squeeze(logits, -1)
                preds = (logits > 0).long()
            else:
                preds = torch.argmax(logits, dim=-1)
        else:
            if self.is_binary:
                preds = (logits > 0).long()
                logits = torch.flatten(logits)
            else:
                preds = torch.argmax(logits, dim=-1)
            preds = torch.flatten(preds)

        return logits, preds, enc_outputs

    @property
    def output_size(self) -> int:
        r"""The feature size of :meth:`forward` output :attr:`logits`.
        If :attr:`logits` size is only determined by input
        (i.e. if ``num_classes`` == 1), the feature size is equal to ``-1``.
        Otherwise it is equal to last dimension value of :attr:`logits` size.
        """
        if self._hparams.num_classes == 1:
            logit_dim = -1
        elif self._hparams.num_classes > 1:
            logit_dim = self._hparams.num_classes
        elif self._hparams.clas_strategy == 'all_time':
            logit_dim = (self._encoder.output_size *
                         self._hparams.max_seq_length)
        elif self._hparams.clas_strategy == 'cls_time':
            logit_dim = self._encoder.output_size
        elif self._hparams.clas_strategy == 'time_wise':
            logit_dim = self._hparams.encoder.dim

        return logit_dim

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

"""
Our basic bi-LSTM tagging model is a context bi-LSTM taking as input word embeddings~w.
We compute subtoken-level (either characters ~c or unicode byte ~b) embed-dings of words using a sequence bi-LSTM at the
This representation is then concate-nated with the (learned) word embeddings vector ~w which forms the input to the context bi-LSTM atthe next layer

For all models we use the same hyperparameters,which  were  set  on  English  dev,  i.e.,

SGD  train-ing with cross-entropy loss,
no mini-batches,
20 epochs
default learning rate (0.1),
128 dimensions for word embeddings,
100 for character and byte embeddings,
100 hidden states
Gaussian noise with σ=0.2.

use a fixed seed throughout

Embeddings are not initialized  with  pre-trained  embeddings,  except when reported otherwise.
In that case we use off-the-shelf polyglot embeddings

For languages with token segmen-tation  ambiguity  we  use  the  provided  gold  seg-mentation.
If  there  is  more  than  one  treebank per  language,  we  use  the  treebank  that  has  the canonical language name (e.g.,Finnish instead ofFinnish-FTB).

The  bi-LSTM  model  performs  already  surpris-ingly well after only 500 training sentences

We  investigated  the  susceptibility of the models to noise,  by artificially corrupting training labels.
"""


class PosTagger(nn.Module):
    def __init__(
        self,
        n_subtoken_embeddings,
        subtoken_embedding_dim,
        n_byte_embeddings,
        byte_embedding_dim,
        n_word_embeddings,
        word_embedding_dim,
        hidden_size,
        n_out,
        n_freq_bins,
        noise_sd,
        use_word,
        use_char,
        use_byte,
        pretrained_embedding
    ):
        assert any([use_word, use_char, use_byte]), "you have to give some input to the lstm"
        super().__init__()
        self.use_pretrained_embedding = pretrained_embedding is not None
        self.use_word = use_word
        self.use_byte = use_byte
        self.use_char = use_char
        self.subtoken_embeddings = nn.Embedding(
            n_subtoken_embeddings, subtoken_embedding_dim
        )
        self.subtoken_lstm = nn.LSTM(
            input_size=subtoken_embedding_dim,
            hidden_size=int(subtoken_embedding_dim/2),
            bidirectional=True,
        )
        self.byte_embeddings = nn.Embedding(
            n_byte_embeddings, byte_embedding_dim
        )
        self.byte_lstm = nn.LSTM(
            input_size=byte_embedding_dim,
            hidden_size=int(byte_embedding_dim/2),
            bidirectional=True,
        )

        word_embedding_dim = word_embedding_dim if pretrained_embedding is None else pretrained_embedding.shape[1]
        self.word_embeddings = nn.Embedding(n_word_embeddings, word_embedding_dim)
        if pretrained_embedding is not None:
            self.word_embeddings.weight.data = pretrained_embedding

        lstm_input_size = 0
        if use_word:
            lstm_input_size += word_embedding_dim
        if use_char:
            lstm_input_size += subtoken_embedding_dim
        if use_byte:
            lstm_input_size += byte_embedding_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=2,
        )
        self.output_projection = nn.Linear(hidden_size * 2, n_out)
        self.output_projection_freq_bin = nn.Linear(hidden_size * 2, n_freq_bins)
        self.noise_sd = noise_sd
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.subtoken_embeddings.weight.data.uniform_(-initrange, initrange)
        self.byte_embeddings.weight.data.uniform_(-initrange, initrange)
        if not self.use_pretrained_embedding:
            self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.output_projection.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()

    def forward(self, X, return_freq_bins):
        # so far no batching, X is a dictionary with keys "chars" and "tokens"
        # the paper doesn't use mini batches so it's ok

        padded_chars = pad_sequence(
            [torch.tensor(t, dtype=torch.long) for t in X["chars"]]
        )
        char_embeds = self.subtoken_embeddings(padded_chars)
        output_char, (h_n_char, c_n_char) = self.subtoken_lstm(char_embeds)
        char_reps = h_n_char.permute(1, 0, 2).reshape(h_n_char.shape[1], -1)

        padded_byte = pad_sequence(
            [torch.tensor(t, dtype=torch.long) for t in X["bytes"]]
        )
        byte_embeds = self.byte_embeddings(padded_byte)
        output_byte, (h_n_byte, c_n_byte) = self.byte_lstm(byte_embeds)
        byte_reps = h_n_byte.permute(1, 0, 2).reshape(h_n_byte.shape[1], -1)

        word_reps = self.word_embeddings(torch.tensor(X["tokens"], dtype=torch.long))
        _combined_reps = []
        if self.use_byte:
            _combined_reps.append(byte_reps)
        if self.use_char:
            _combined_reps.append(char_reps)
        if self.use_word:
            _combined_reps.append(word_reps)
        combined_reps = torch.cat(_combined_reps, axis=1)[None, :, :]
        #combined_reps = word_reps[None, :, :]

        output, (h_n, c_n) = self.lstm(combined_reps)

        rval = self.output_projection(output)
        if not return_freq_bins:
            return rval + (torch.randn_like(rval) * self.noise_sd)
        rval_freq_bin = self.output_projection_freq_bin(output)
        return F.log_softmax(rval + (torch.randn_like(rval) * self.noise_sd), dim=2), F.log_softmax(rval_freq_bin, dim=2)

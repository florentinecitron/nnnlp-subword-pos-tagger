import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from lstm import LSTM

torch.manual_seed(111)


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
        if pretrained_embedding is not None and word_embedding_dim < pretrained_embedding.shape[1]:
            raise AssertionError("word_embedding_dim must be equal to or greater than pretrained embedding dim")
        super().__init__()
        # where unknown tokens should use random embeddings
        self.unk_to_random_embed = False
        self.use_pretrained_embedding = pretrained_embedding is not None
        self.use_word = use_word
        self.use_byte = use_byte
        self.use_char = use_char
        if self.use_char:
            self.subtoken_embeddings = nn.Embedding(
                n_subtoken_embeddings, subtoken_embedding_dim, padding_idx=0
            )
            self.subtoken_lstm = nn.LSTM(
                input_size=subtoken_embedding_dim,
                hidden_size=subtoken_embedding_dim,
                bidirectional=True,
                num_layers=1,
                #dropout=.2
            )
        if self.use_byte:
            self.byte_embeddings = nn.Embedding(
                n_byte_embeddings, byte_embedding_dim, padding_idx=0
            )
            self.byte_lstm = nn.LSTM(
                input_size=byte_embedding_dim,
                hidden_size=byte_embedding_dim,
                bidirectional=True,
                num_layers=1,
                #dropout=.2
            )

        if self.use_word:
            if pretrained_embedding is not None:
                word_embedding_dim = pretrained_embedding.shape[1]
            self.word_embeddings = nn.Embedding(n_word_embeddings, word_embedding_dim)
            if pretrained_embedding is not None:
                self.word_embeddings.weight.data = pretrained_embedding.clone()

        lstm_input_size = 0
        if use_word:
            lstm_input_size += word_embedding_dim
        if use_char:
            lstm_input_size += subtoken_embedding_dim * 2
        if use_byte:
            lstm_input_size += byte_embedding_dim * 2

        self.lstm = nn.LSTM (
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            bidirectional=True
            #dropout=.2
        )
        self.output_projection = nn.Linear(hidden_size * 2, n_out)
        #self.output_projection = nn.Sequential(nn.Linear(hidden_size * 2, 50), nn.Tanh(), nn.Linear(50, n_out))
        self.output_projection_freq_bin = nn.Linear(hidden_size * 2, n_freq_bins)
        self.noise_sd = noise_sd

    def forward(self, X, return_freq_bins, train=True):
        # so far no batching, X is a dictionary with keys "chars" and "tokens"
        # the paper doesn't use mini batches so it's ok

        if self.use_char:
            padded_chars = pad_sequence(
                [torch.tensor(t, dtype=torch.long) for t in X["chars"]],
                padding_value=0
            )
            char_embeds = self.subtoken_embeddings(padded_chars)
            _, (h_n_char, _) = self.subtoken_lstm(char_embeds)
            char_reps = h_n_char.permute(1, 0, 2).reshape(h_n_char.shape[1], -1)

        if self.use_byte:
            padded_byte = pad_sequence(
                [torch.tensor(t, dtype=torch.long) for t in X["bytes"]],
                padding_value=0
            )
            byte_embeds = self.byte_embeddings(padded_byte)
            _, (h_n_byte, _) = self.byte_lstm(byte_embeds)
            byte_reps = h_n_byte.permute(1, 0, 2).reshape(h_n_byte.shape[1], -1)

        if self.use_word:
            word_reps = self.word_embeddings(torch.tensor(X["tokens"], dtype=torch.long))
            if self.unk_to_random_embed:
                word_reps[X["unk_indices"], :] = torch.randn_like(word_reps[X["unk_indices"], :])

        _combined_reps = []
        if self.use_byte:
            _combined_reps.append(byte_reps)
        if self.use_char:
            _combined_reps.append(char_reps)
        if self.use_word:
            _combined_reps.append(word_reps)
        _combined_reps = torch.cat(_combined_reps, axis=1)[None, :, :]

        if train:
            combined_reps = _combined_reps + (torch.randn_like(_combined_reps) * self.noise_sd)
        else:
            combined_reps = _combined_reps

        output = torch.tanh(self.lstm(combined_reps)[0]).permute(1,0,2)

        if train:
            rval = self.output_projection(output + (torch.randn_like(output) * self.noise_sd))
        else:
            rval = self.output_projection(output)

        if not return_freq_bins:
            return rval
        if train:
            rval_freq_bin = self.output_projection_freq_bin(output + (torch.randn_like(output) * self.noise_sd))
        else:
            rval_freq_bin = self.output_projection_freq_bin(output)

        return rval, rval_freq_bin
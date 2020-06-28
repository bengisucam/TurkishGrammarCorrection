import numpy
import torch.nn as nn
import torch
import torch.nn.functional as F


class BiLSTM(nn.Module):
    def __init__(self, hidden_size, embed_size, layers, chars, src, lstm_dropout=0.2):
        super(BiLSTM, self).__init__()

        self.chars = chars
        self.src = src

        self.vocab = self.chars.vocab
        char_size = len(self.chars.vocab)

        self.hidden_size = hidden_size
        self.char_embedding = nn.Embedding(char_size, embed_size)
        torch.nn.init.xavier_uniform_(self.char_embedding.weight)

        self.char_embedding.weight.requires_grad = True

        self.rnn = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True, dropout=lstm_dropout,
                           num_layers=layers)

    def tokenize(self, sentence):
        return self.chars.tokenize(sentence)

    def forward(self, input):

        char_embeds = self.char_embedding(input)
        output, _ = self.rnn(char_embeds)

        output = output[:, output.size(1) - 1]
        return output

    def _word2charIndices(self, word):
        w = self.src.vocab.itos[word]
        tokens = self.tokenize(w)
        return torch.tensor([self.chars.vocab.stoi[i] for i in tokens])

    def encode(self, batches):
        w = torch.empty((len(batches), batches.size(1), self.hidden_size * 2), dtype=torch.float)
        for i in range(len(batches)):
            # for each sentence
            sentence_tensor = batches[i]
            char_indices = self._sentencetensor2charIndices(sentence_tensor)
            embeds = self.forward(char_indices)
            w[i] = embeds
        return w.cuda()

    def __call__(self, input_variables):
        return self.encode(input_variables)

    def _sentencetensor2charIndices(self, sentence):
        max_char_size = 21
        w = torch.empty((len(sentence), max_char_size), dtype=torch.long)
        index = 0
        for wordIndex in sentence:
            chars = self._word2charIndices(wordIndex)
            result = F.pad(input=chars, pad=(0, max_char_size - len(chars)), mode='constant',
                           value=self.chars.vocab.stoi[self.chars.pad_token])
            w[index] = result
            index += 1
        return w.cuda()

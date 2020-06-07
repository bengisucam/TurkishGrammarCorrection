import torch.nn as nn
import torch


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,char_size, layers, lstm_dropout):
        super(BiLSTM, self).__init__()
        self.char_embedding = nn.Embedding(char_size, hidden_size)
        self.char_embedding.weight.requires_grad=True

        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True, dropout=lstm_dropout,
                           num_layers=layers)

    def forward(self, input, input_lengths=None):
        embedded = self.embedding(input)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def one_hot_encode(self,input):
        pass
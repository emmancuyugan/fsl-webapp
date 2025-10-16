import torch.nn as nn

class LSTMGRUHybrid(nn.Module):
    def __init__(self, input_size: int, num_classes: int,
                 lstm_hidden: int = 128,
                 gru_hidden:  int = 128,
                 fc_hidden:   int = 160,
                 dropout_p:   float = 0.20):
        super().__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden, num_layers=1, batch_first=True)
        self.bn_lstm = nn.LayerNorm(lstm_hidden)
        self.drop_lstm = nn.Dropout(dropout_p)

        self.gru  = nn.GRU(lstm_hidden, gru_hidden, num_layers=1, batch_first=True)
        self.bn_gru = nn.LayerNorm(gru_hidden)
        self.drop_gru = nn.Dropout(dropout_p)

        self.fc1 = nn.Linear(gru_hidden, fc_hidden)
        self.bn_fc1 = nn.LayerNorm(fc_hidden)
        self.drop_fc1 = nn.Dropout(dropout_p)

        self.fc_out = nn.Linear(fc_hidden, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.bn_lstm(lstm_out)
        lstm_out = self.drop_lstm(lstm_out)

        gru_out, h_n = self.gru(lstm_out)
        h_last = h_n[-1]
        h_last = self.bn_gru(h_last)
        h_last = self.drop_gru(self.act(h_last))

        z = self.fc1(h_last)
        z = self.bn_fc1(z)
        z = self.drop_fc1(self.act(z))
        return self.fc_out(z)

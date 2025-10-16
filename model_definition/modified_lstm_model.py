import torch
import torch.nn as nn

class ModifiedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 dropout=0.30, use_batch_norm=False, use_layernorm=True):
        super(ModifiedLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_batch_norm = use_batch_norm
        self.use_layernorm = use_layernorm

        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size if i == 0 else hidden_size,
                hidden_size,
                batch_first=True,
                dropout=0.0
            )
            for i in range(num_layers)
        ])

        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(hidden_size) for _ in range(num_layers)
            ])
        if use_layernorm:
            self.layernorms = nn.ModuleList([
                nn.LayerNorm(hidden_size) for _ in range(num_layers)
            ])

        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, sequence, reset_mask=None):
        batch_size, seq_len, _ = sequence.size()
        h = [
            torch.zeros(1, batch_size, self.hidden_size, device=sequence.device)
            for _ in range(self.num_layers)
        ]
        c = [
            torch.zeros(1, batch_size, self.hidden_size, device=sequence.device)
            for _ in range(self.num_layers)
        ]

        out = sequence
        for layer_index, lstm in enumerate(self.lstm_layers):
            outputs = []
            h_t = h[layer_index]
            c_t = c[layer_index]

            for t in range(seq_len):
                input_t = out[:, t:t+1, :]
                _, (h[layer_index], c[layer_index]) = lstm(input_t, (h_t, c_t))

                if reset_mask is not None:
                    mask = reset_mask[:, t].view(1, batch_size, 1)
                    h[layer_index] = h[layer_index] * mask
                    c[layer_index] = c[layer_index] * mask

                outputs.append(h[layer_index].squeeze(0))
                h_t = h[layer_index]
                c_t = c[layer_index]

            out = torch.stack(outputs, dim=1)

            if hasattr(self, "batch_norms"):
                out = self.batch_norms[layer_index](
                    out.reshape(-1, self.hidden_size)
                ).reshape(batch_size, seq_len, self.hidden_size)
            if hasattr(self, "layernorms"):
                out = self.layernorms[layer_index](out)

            out = self.relu(out)

        out = out[:, -1, :]
        return self.fc(out)

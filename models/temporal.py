import torch
import torch.nn as nn

class TemporalFER(nn.Module):
    """
    Video-based FER model using LSTM or Transformer on top of frame features.
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, num_classes=7, architecture='lstm'):
        super(TemporalFER, self).__init__()
        self.architecture = architecture
        
        if architecture == 'lstm':
            self.temporal_net = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True
            )
            self.fc = nn.Linear(hidden_dim * 2, num_classes) # *2 for bidirectional
            
        elif architecture == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, batch_first=True)
            self.temporal_net = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(input_dim, num_classes)
            # Use a class token or average pooling? Let's use average pooling for simplicity.
            
        else:
            raise ValueError("Architecture must be 'lstm' or 'transformer'")

    def forward(self, x):
        """
        Args:
            x: (Batch, Sequence_Length, Input_Dim) tensor of features.
        """
        if self.architecture == 'lstm':
            # LSTM output: (Batch, Seq, Hidden*2)
            output, (hn, cn) = self.temporal_net(x)
            # Use the last hidden state or average?
            # Usually last state for classification.
            # hn is (Num_Layers * Num_Directions, Batch, Hidden)
            # We take the last layer's forward and backward states.
            # Or just take the last time step output.
            last_output = output[:, -1, :]
            logits = self.fc(last_output)
            
        elif self.architecture == 'transformer':
            # Transformer output: (Batch, Seq, Input_Dim)
            output = self.temporal_net(x)
            # Global Average Pooling over time
            avg_pool = torch.mean(output, dim=1)
            logits = self.fc(avg_pool)
            
        return logits

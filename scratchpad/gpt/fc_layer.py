import torch
import torch.nn as nn

# Define the FC layer
fc_layer = nn.Linear(in_features=10, out_features=5)

# Create some input data
input_data = torch.randn(1, 10)
print(input_data)

# Pass the input data through the FC layer
fc_out = fc_layer(input_data)

print(fc_out)
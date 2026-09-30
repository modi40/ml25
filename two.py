import torch
import torch.nn as nn

class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

input_features = 4
hidden_neurons = 6
output_neurons = 1

model = TwoLayerNN(input_features, hidden_neurons, output_neurons)
print("Two Layered Neural Network (ReLU and Sigmoid):")
print(model)

sample_inputs = torch.tensor([[0.5, -1.2, 2.1, 0.8], [1.1, 0.4, -0.7, 1.5]], dtype=torch.float32)
outputs = model(sample_inputs)

print("\nSample Input Tensor:")
print(sample_inputs)
print("\nModel Outputs (Sigmoid Activated):")
print(outputs)

import torch
import torch.nn as nn

# Simple model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2)
)

# Storage for captured values
activations = {}
gradients = {}

# Forward hook: captures layer outputs
def forward_hook(module, input, output):
    """Called after forward pass of this layer"""
    layer_name = module.__class__.__name__
    print(f"[Forward Hook] {layer_name} output shape: {output.shape}")
    activations[layer_name] = output.detach()

# Backward hook: captures gradients
def backward_hook(module, grad_input, grad_output):
    """Called during backward pass of this layer"""
    layer_name = module.__class__.__name__
    print(f"[Backward Hook] {layer_name} gradient shape: {grad_output[0].shape}")
    gradients[layer_name] = grad_output[0].detach()

# Register hooks on the first linear layer
layer = model[0]
forward_handle = layer.register_forward_hook(forward_hook)
backward_handle = layer.register_full_backward_hook(backward_hook)

# Run forward and backward
x = torch.randn(3, 10)  # batch_size=3, features=10
output = model(x)
loss = output.sum()
loss.backward()

# Now activations and gradients are captured!
print("\nCaptured activations:", {k: v.shape for k, v in activations.items()})
print("Captured gradients:", {k: v.shape for k, v in gradients.items()})

# Cleanup: remove hooks when done
forward_handle.remove()
backward_handle.remove()
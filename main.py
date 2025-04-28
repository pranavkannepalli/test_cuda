import torch

print(torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

x = torch.rand(3, 3, device=device)
y = torch.rand(3, 3, device=device)

z = x + y

print("Tensor x")
print(x)

print("Tensor y")
print(y)

print("Tensor z")
print(z)

print(x.is_cuda)
print(y.is_cuda)
print(z.is_cuda)

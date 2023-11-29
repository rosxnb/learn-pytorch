import torch
import torch.optim as optim


def linear_model(x, W, b):
    return torch.matmul(x, W) + b


W = torch.randn((4, 3), requires_grad=True)
b = torch.randn(3, requires_grad=True)
optimizer = optim.Adam([W, b])

data = torch.randn(10, 4)
targets = torch.randn(10, 3)

for sample, target in zip(data, targets):
    # Clear out the gradients of all Variables 
    # in this optimizer (i.e. W, b)
    optimizer.zero_grad()
    output = linear_model(sample, W, b)
    loss = torch.sum((output - target) ** 2)
    loss.backward()
    optimizer.step()


## Instead, if we were to do vanila gradient descent
# learning_rate = 0.01
# for sample, target in zip(data, targets):
#     # clear out the gradients of Variables 
#     # (i.e. W, b)
#     W.grad.data.zero_()
#     b.grad.data.zero_()
#
#     output = linear_model(sample, W, b)
#     loss = (output - target) ** 2
#     loss.backward()
#
#     W -= learning_rate * W.grad.data
#     b -= learning_rate * b.grad.data

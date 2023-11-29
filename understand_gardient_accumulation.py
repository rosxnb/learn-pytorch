import torch

def single_value():
    print('Single Value')
    x = torch.randn(1, requires_grad=True)
    y = x ** 4

    y.backward()
    print('x gradient after first y.backward(): ', end='')
    print(x.grad)

    dy = 4 * (x ** 3)
    print('manual computation of x grad: ' + str(dy))
    print(x.grad == dy)


def vector_value():
    print('Vector Value')
    x = torch.randn((2, 1), requires_grad=True)
    y = torch.zeros((3, 1))

    y[0] = x[0] ** 2
    y[1] = x[1] ** 2
    y[2] = x[1] ** 4
    y.backward(gradient=torch.ones(y.shape))

    print('x gradient after first y.backward(): ', end='')
    print(x.grad)

    dy0 = 2 * x[0]
    dy1 = 2 * x[1]
    dy2 = 4 * (x[1] ** 3)
    dy = torch.tensor([[dy0], [dy1 + dy2]])
    print('manual computation of x grad: ', end='')
    print(dy)
    print(dy == x.grad)


single_value()
print()
vector_value()

import torch
import torch.nn.functional as F


class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1=torch.randn(linear_1_out_features, linear_1_in_features),
            b1=torch.randn(linear_1_out_features),
            W2=torch.randn(linear_2_out_features, linear_2_in_features),
            b2=torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1=torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1=torch.zeros(linear_1_out_features),
            dJdW2=torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2=torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        self.cache['x'] = x
        s1 = x @ self.parameters["W1"].T + self.parameters["b1"].unsqueeze(0)
        self.cache['s1'] = s1
        z1 = self.non_linearity(s1, self.f_function)
        self.cache['z1'] = z1
        s2 = z1 @ self.parameters["W2"].T + self.parameters["b2"].unsqueeze(0)
        self.cache['s2'] = s2
        z2 = self.non_linearity(s2, self.g_function)

        return z2

    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """

        # -- LINEAR 2
        d_g_function = self.non_linearity_derivative(self.g_function)

        # print(self.cache['z1'].shape, self.cache['s2'].shape,
        #       dJdy_hat.shape, self.parameters['W2'].shape)

        self.grads['dJdW2'] = (
            (self.cache['z1'].T @ (d_g_function(self.cache['s2']) * dJdy_hat))).T

        # print(self.cache['s2'].shape,
        #       dJdy_hat.shape, self.parameters['b2'].shape)

        self.grads['dJdb2'] = (d_g_function(
            self.cache['s2']) * dJdy_hat).sum(0)

        # -- LINEAR 1
        d_f_function = self.non_linearity_derivative(self.f_function)

        self.grads['dJdW1'] = (d_f_function(
            self.cache['s1']) * ((d_g_function(self.cache['s2']) * dJdy_hat) @ self.parameters['W2'])).T @ self.cache['x']

        self.grads['dJdb1'] = (d_f_function(
            self.cache['s1']) * ((d_g_function(self.cache['s2']) * dJdy_hat) @ self.parameters['W2'])).sum(0)

    def non_linearity(self, x, function):
        if function == 'relu':
            return relu(x)
        elif function == 'sigmoid':
            return sigmoid(x)
        else:
            return identity(x)

    def non_linearity_derivative(self, function):
        if function == 'relu':
            return d_relu
        elif function == 'sigmoid':
            return d_sigmoid
        else:
            return d_identity

    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()


def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    m = y.flatten().shape[0]

    loss_scalar = 1/m*((y - y_hat)**2).sum()
    dJdy_hat = -2/m*(y - y_hat)

    return loss_scalar, dJdy_hat

    # return loss, dJdy_hat


def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor

    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    m = y.flatten().shape[0]

    loss_scalar = -1/m*(y*torch.log(y_hat) + (1-y)*torch.log(1-y_hat)).sum()
    dJdy_hat = -1/m*(y/y_hat - (1 - y)/(1-y_hat))

    return loss_scalar, dJdy_hat


def relu(x):
    return torch.where(x > 0, x, torch.tensor(0, dtype=x.dtype))


def d_relu(x):
    return torch.where(x > 0, torch.tensor(1, dtype=x.dtype), torch.tensor(0, dtype=x.dtype))


def sigmoid(x):
    return 1/(1 + torch.exp(-x))


def d_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))


def identity(x):
    return x


def d_identity(x):
    return torch.ones_like(x)


if __name__ == "__main__":
    mlp = MLP(
        linear_1_in_features=2,
        linear_1_out_features=20,
        f_function='relu',
        linear_2_in_features=20,
        linear_2_out_features=5,
        g_function='identity'
    )

    t = torch.randn(10, 2)
    y = torch.randn(10, 5)

    # print(mlp.forward(t).shape)
    x = mlp.forward(t)

    loss, jacobian = mse_loss(y, x)
    mlp.backward(jacobian)

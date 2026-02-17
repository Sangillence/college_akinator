import tensorflow as tf

class TFQNetwork(tf.Module):
    def __init__(self, state_size, action_size, pytorch_model):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Extract PyTorch weights and convert to TF variables
        self.w1 = tf.Variable(pytorch_model.fc1.weight.detach().numpy().T)
        self.b1 = tf.Variable(pytorch_model.fc1.bias.detach().numpy())
        self.w2 = tf.Variable(pytorch_model.fc2.weight.detach().numpy().T)
        self.b2 = tf.Variable(pytorch_model.fc2.bias.detach().numpy())
        self.w3 = tf.Variable(pytorch_model.fc3.weight.detach().numpy().T)
        self.b3 = tf.Variable(pytorch_model.fc3.bias.detach().numpy())

        # Wrap forward pass in a tf.function with input_signature
        self.forward = tf.function(self._forward,
                                   input_signature=[tf.TensorSpec([1, self.state_size], tf.float32)])

    def _forward(self, x):
        x = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        x = tf.nn.relu(tf.matmul(x, self.w2) + self.b2)
        x = tf.matmul(x, self.w3) + self.b3
        return x

    def __call__(self, x):
        return self.forward(x)

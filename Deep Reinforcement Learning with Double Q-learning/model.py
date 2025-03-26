import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class GenericQNetwork(nn.Module):

    def __init__(self, input_size, output_size, channels):
        super().__init__()
        # first convolutional layer values
        self.nb_filters_l1 = 32
        self.size_filters_l1 = 8
        self.stride_l1 = 4

        # second convolutional layer values
        self.nb_filters_l2 = 64
        self.size_filters_l2 = 4
        self.stride_l2 = 2

        # first linear layer values
        self.ffn1 = 512

        # output linear layer values
        self.nb_actions = output_size
        
        self.nb_channels = channels
        self.image_size = input_size

        self.conv_layer1 = nn.Parameter(torch.randn((self.nb_filters_l1, self.nb_channels, self.size_filters_l1, self.size_filters_l1), dtype=torch.float32), requires_grad=True)
        self.bias_layer1 = nn.Parameter(torch.zeros((self.nb_filters_l1,), dtype=torch.float32), requires_grad=True)
        init.kaiming_normal_(self.conv_layer1, nonlinearity="relu")
        self.conv_layer2 = nn.Parameter(torch.randn((self.nb_filters_l2, self.nb_filters_l1, self.size_filters_l2, self.size_filters_l2), dtype=torch.float32), requires_grad=True)
        self.bias_layer2 = nn.Parameter(torch.zeros((self.nb_filters_l2,), dtype=torch.float32), requires_grad=True)
        init.kaiming_normal_(self.conv_layer2, nonlinearity="relu")

        test_input = torch.randn(1, self.nb_channels, self.image_size, self.image_size)
        with torch.no_grad():
            self.conv1_output = F.conv2d(test_input, self.conv_layer1, stride=self.stride_l1)
            self.conv2_output = F.conv2d(self.conv1_output, self.conv_layer2, stride=self.stride_l2)
            _, _, height, width = self.conv2_output.size()
            self.ffn_size = self.nb_filters_l2 * height * width

        self.ffn_layer = nn.Parameter(torch.randn((self.ffn_size, self.ffn1), dtype=torch.float32), requires_grad=True)
        init.kaiming_normal_(self.ffn_layer, nonlinearity="relu")
        self.ffn_bias = nn.Parameter(torch.zeros((1, self.ffn1), dtype=torch.float32), requires_grad=True)

        self.output_layer = nn.Parameter(torch.randn((self.ffn1, self.nb_actions), dtype=torch.float32), requires_grad=True)
        # init.kaiming_normal_(self.output_layer, nonlinearity="relu")
        self.output_bias = nn.Parameter(torch.zeros((1, self.nb_actions), dtype=torch.float32), requires_grad=True)

        # optimizer
        self.optimizer = torch.optim.Adam(
            params=[
                self.conv_layer1,
                self.bias_layer1,
                self.conv_layer2,
                self.bias_layer2,
                self.ffn_layer,
                self.ffn_bias,
                self.output_layer,
                self.output_bias,
            ],
            lr=0.00025,
        )
    
    def forward(self, x):
        x = F.conv2d(x, self.conv_layer1, bias=self.bias_layer1, stride=self.stride_l1)
        x = F.relu(x)
        x = F.conv2d(x, self.conv_layer2, bias=self.bias_layer2, stride=self.stride_l2)
        x = F.relu(x)
        x = x.view(-1, self.ffn_size)
        x = torch.matmul(x, self.ffn_layer) + self.ffn_bias
        x = F.relu(x)
        x = torch.matmul(x, self.output_layer) + self.output_bias
        return x
    
    def action_selection(self, epsilon, logits):
        if torch.rand(1) < epsilon:
            return torch.randint(0, self.nb_actions, (logits.size()[0], 1))
        else:
            with torch.no_grad():
                return torch.argmax(logits, dim=1, keepdim=True)

    def values_from_actions(self, logits, actions):
        return logits.gather(1, actions)
    

class QNetwork(GenericQNetwork):

    def __init__(self, input_size, output_size, channels):
        super().__init__(input_size, output_size, channels)

class QTargetNetwork(GenericQNetwork):

    def __init__(self, input_size, output_size, channels):
        super().__init__(input_size, output_size, channels)
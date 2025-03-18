import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import CustomLRScheduler
import torch.nn.init as init

class GenericNetwork(nn.Module):

    def __init__(self, stack_param, batch_size):
        super().__init__()
        self.batch_size = batch_size
        image_size = 84 # assuming squared image
        self.nb_filters_l1 = 16
        self.nb_filters_l2 = 32
        self.filter_size_l1 = 8
        self.filter_size_l2 = 4
        self.ffn_size = 256
        self.nb_actions = 3

        self.filters_layer1 = nn.Parameter(
            torch.randn((self.nb_filters_l1, stack_param, self.filter_size_l1, self.filter_size_l1), dtype=torch.float32, requires_grad=True)# * 0.01
        )
        self.filters_layer2 = nn.Parameter(
            torch.randn((self.nb_filters_l2, self.nb_filters_l1, self.filter_size_l2, self.filter_size_l2), dtype=torch.float32, requires_grad=True)# * 0.01
        )
        init.kaiming_normal_(self.filters_layer1, nonlinearity="relu")
        init.kaiming_normal_(self.filters_layer2, nonlinearity="relu")

        self.bias_layer1 = nn.Parameter(
            torch.zeros(self.nb_filters_l1, requires_grad=True)
        )
        self.bias_layer2 = nn.Parameter(
            torch.zeros(self.nb_filters_l2, requires_grad=True)
        )

        test_input = torch.zeros((1, stack_param, image_size, image_size))
        with torch.no_grad():
            test_l1 = F.conv2d(test_input, self.filters_layer1, stride=4)
            test_l2 = F.conv2d(test_l1, self.filters_layer2, stride=2)
            _, _, height, width = test_l2.size()
            flatten_size = self.nb_filters_l2 * height * width

        # fan_in = flatten_size
        bound = 1 / flatten_size
        self.linear_layer = nn.Parameter(
            torch.randn((flatten_size, self.ffn_size), dtype=torch.float32, requires_grad=True)# * bound * 0.001
        )
        init.kaiming_normal_(self.linear_layer, nonlinearity="relu")

        self.linear_bias = nn.Parameter(torch.zeros((1, self.ffn_size), dtype=torch.float32, requires_grad=True))

        self.final_layer = nn.Parameter(
            torch.randn((self.ffn_size, self.nb_actions), dtype=torch.float32, requires_grad=True)# * 0.01
        )
        self.final_bias = nn.Parameter(torch.zeros((1, self.nb_actions), dtype=torch.float32, requires_grad=True))
        self.optimizer = torch.optim.AdamW(
            params=[
                self.filters_layer1,
                self.bias_layer1,
                self.filters_layer2,
                self.bias_layer2,
                self.linear_layer,
                self.linear_bias,
                self.final_layer,
                self.final_bias,
            ],
            lr=0.0005,
        )
        self.scheduler = CustomLRScheduler(
            self.optimizer,
            factor=0.9,  # decrease by 10% when loss plateaus
            increase_factor=1.1,  # increase by 10% when loss improves
            patience=2000,
            min_lr=5e-7,
            max_lr=1e-3,
            verbose=True
        )


    def forward(self, x):
        output_layer1 = F.relu(F.conv2d(x, self.filters_layer1, stride=4) + self.bias_layer1.view(1, -1, 1, 1))
        output_layer2 = F.relu(F.conv2d(output_layer1, self.filters_layer2, stride=2) + self.bias_layer2.view(1, -1, 1, 1))
        flatten_output_layer2 = output_layer2.view(output_layer2.size()[0], -1)
        flatten_layer2 = F.relu(torch.matmul(flatten_output_layer2, self.linear_layer) + self.linear_bias)
        self.action_prediction = torch.matmul(flatten_layer2, self.final_layer) + self.final_bias
        return self.action_prediction
    
    def action_selection(self, epsilon: float = 0.1):
        random_selection = 0 if random.random() < 1 - epsilon else 1
        if not random_selection: # max prediction
            action = torch.argmax(self.action_prediction, dim=1)
        else:
            action = torch.randint(0, self.nb_actions, (self.batch_size,1))
        return action
    
    def logits_from_actions(self, actions):
        return self.action_prediction.gather(1, actions)


class QNetwork(GenericNetwork):

    def __init__(self, stack_param, batch_size=1):
        super().__init__(stack_param, batch_size)

    
class QTargetNetwork(GenericNetwork):

    def __init__(self, stack_param, batch_size=1):
        super().__init__(stack_param, batch_size)

# test example
if __name__ == "__main__":

    qnetwork = QNetwork(4, 6)
    example_image = torch.randn((6, 4, 336, 336), dtype=torch.float32)
    output = qnetwork.forward(example_image)
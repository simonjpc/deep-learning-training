import torch
import torch.nn.functional as F

class GenericNetwork:

    def __init__(self, stack_param):
        self.batch_size = 32
        self.nb_filters_l1 = 16
        self.nb_filters_l2 = 32
        self.filter_size_l1 = 8
        self.filter_size_l2 = 4
        self.ffn_size = 256
        self.nb_actions = 3
        self.out_height2 = None # placeholder
        self.out_width2 = None # placeholder
        self.filters_layer1 = torch.randn((self.nb_filters_l1, stack_param, self.filter_size_l1, self.filter_size_l1), dtype=torch.float32) * 0.01
        self.filters_layer2 = torch.randn((self.nb_filters_l2, self.nb_filters_l1, self.filter_size_l2, self.filter_size_l2), dtype=torch.float32) * 0.01
        
        self.final_layer = torch.randn((self.ffn_size, self.nb_actions), dtype=torch.float32) * 0.01

    def forward(self, x):
        
        output_layer1 = F.conv2d(x, self.filters_layer1, stride=4)
        output_layer2 = F.conv2d(output_layer1, self.filters_layer2, stride=2)
        _, _, height, width = output_layer2.size()
        self.out_height2 = height
        self.out_width2 = width
        flatten_output_layer2 = output_layer2.view(-1, self.nb_filters_l2 * self.out_height2 * self.out_width2)
        self.linear_layer = torch.randn((int(flatten_output_layer2.size()[-1]), self.ffn_size), dtype=torch.float32) * 0.01
        output_layer2 = flatten_output_layer2 @ self.linear_layer
        action_prediction = output_layer2 @ self.final_layer
        return action_prediction


class QNetwork(GenericNetwork):

    def __init__(self, stack_param):
        super().__init__(stack_param)

    
class QTargetNetwork(GenericNetwork):

    def __init__(self):
        super().__init__()

# test example
if __name__ == "__main__":

    qnetwork = QNetwork(4)
    example_image = torch.randn((6, 4, 336, 336), dtype=torch.float32)
    output = qnetwork.forward(example_image)
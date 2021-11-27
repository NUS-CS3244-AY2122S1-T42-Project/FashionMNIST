import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Visualizer():
    def __init__(self, model):
        self.model = model
        self.model_weights = []
        self.conv_layers = []
    
    def get_num_conv_layers(self):
        model_children = list(self.model.children())
        count = 0

        # append all the conv layers and their respective weights to the list
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                count += 1
                self.model_weights.append(model_children[i].weight)
                self.conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for child in model_children[i].children():
                    if type(child) == nn.Conv2d:
                        count += 1
                        self.model_weights.append(child.weight)
                        self.conv_layers.append(child)
        print(f"Total convolutional layers: {count}")
    
    def visualize_layers_and_weights(self):
        # take a look at the conv layers and the respective weights
        for weight, conv in zip(self.model_weights, self.conv_layers):
            # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
            print(f"CONV: {conv} ====> SHAPE: {weight.shape}")
    
    def visualize_and_save_filter(self, layer=None):
        layers = range(len(self.conv_layers))
        if layer:
            layers = [layer]
        
        for num_layer in layers:
            # visualize the first conv layer filters
            plt.figure(figsize=(20, 17))
            for i, filter in enumerate(self.model_weights[num_layer].cpu()):
                plt.subplot(8, 8, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
                plt.imshow(filter[0, :, :].detach(), cmap='gray')
                plt.axis('off')
                plt.savefig('./diagrams/filter2.png')
            plt.show()
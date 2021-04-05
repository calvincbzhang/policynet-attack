import argparse
import torch

import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from deepfool import deepfool

import sys
sys.path.append('../')
from mnist_classification.net import Net

def parse_args():
    parser = argparse.ArgumentParser()

    # show a few examples of attacked images
    parser.add_argument('--show', action='store_true')  # False by default, if --show then True
    # attack on entire dataset
    parser.add_argument('--full', action='store_true')  # False by default, if --full then True

    return parser.parse_args()


def main(args):
    # check for CUDA enabled GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))
    ])

    testset = datasets.MNIST('../../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=True)

    model = Net()
    model.load_state_dict(torch.load("../../models/mnist_classification.pt"))
    model.to(device)

    if args.show:
        # get some random test images
        dataiter = iter(testloader)
        images, labels = dataiter.next()
        images = images.to(device)
        labels = labels.to(device)

        # show some perturbed image with truth and predicted
        fig = plt.figure()

        for i in range(8):
            plt.subplot(3, 4, i+1)
            plt.tight_layout()
            r_tot, loop_i, label, k_i, pert_image = deepfool(images.cpu()[i], model)
            plt.imshow(pert_image.cpu()[0, 0], cmap='gray', interpolation='none')
            plt.title(f'True: {labels[i]}, Adv: {k_i}')
            plt.xticks([])
            plt.yticks([])
        
        plt.show()

    if args.full:
        correct = 0
        dataiter = iter(testloader)
        while True:
            try:
                images, labels = dataiter.next()
            except StopIteration:
                break  # Iterator exhausted: stop the loop
            else:
                for i in range(len(images)):
                    _, _, _, k_i, _ = deepfool(images[i], model)
                    if labels[i] == k_i:
                        correct += 1
                
        print(f'\nCorrect: {correct}\n')
        print(f'Total: {len(testloader.dataset)}\n')
        print(f'Percentage correct: {100 * correct / len(testloader.dataset)}%')


if __name__ == "__main__":
    args = parse_args()
    main(args)
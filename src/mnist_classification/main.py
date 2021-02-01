import argparse
import torch
import torchvision

import torch.optim as optim
import torch.nn.functional as F

from net import Net
from torchvision import datasets, transforms

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=bool, default=False)
    # LR and momentum for SGD
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    # number of epochs for training
    parser.add_argument('--epochs', type=int, default=10)

    return parser.parse_args()


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(args):
    # check for CUDA enabled GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))
    ])

    trainset = datasets.MNIST('../../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.MNIST('../../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=True)

    # instantiate net
    net = Net().to(device)

    if args.train:
        # train net
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)

        for epoch in range(args.epochs):
            train(net, device, trainloader, optimizer, epoch)
            test(net, device, testloader)

        # save model
        torch.save(net.state_dict(), "../../models/mnist_classification.pt")
    else:
        # load model and test
        net.load_state_dict(torch.load("../../models/mnist_classification.pt"))
        net.to(device)
        test(net, device, testloader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
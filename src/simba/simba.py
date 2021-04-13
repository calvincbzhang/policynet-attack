'''
Code inspired from the official repository https://github.com/cg563/simple-blackbox-attack
'''

import torch
import cv2
import matplotlib.pyplot as plt
import logging as log

def simba(x, y, model, num_iters=10000, epsilon=0.1):

    num_pixels = x.reshape(1, -1).size(1)
    # get a random permutation of of the numbers from 0 to num_pixels-1
    perm = torch.randperm(num_pixels)

    # get the q_value from the model
    q_val = model(x.to('cuda'))[0, y].view(1,1)

    attack = torch.zeros(num_pixels)
    attack = attack.view(x.size())

    new_y = y
    i = 0

    # while the predictions are the same, keep trying
    while new_y == y and i < num_iters:

        delta = torch.zeros(num_pixels)
        delta[perm[i]] = epsilon
        delta = delta.view(x.size())

        # try adding and subtracting epsilon * q = delta
        for e in {1, -1}:
            new_x = x + epsilon * delta

            # compute the q_value for x + delta +/- epsilon*q
            new_q_val = model(new_x.to('cuda')).max(1)[0].view(1,1)
            new_y = model(new_x.to('cuda')).max(1)[1].view(1,1)

            # if the updates q_value decreased
            if new_q_val < q_val:
                q_val = new_q_val
                x = new_x
                attack += (epsilon * delta)
        i += 1
    
    print(f'Iterations for SimBA: {i}')
    log.info(f'Iterations for SimBA: {i}')
    # plt.imshow(new_x.to('cpu').reshape(1, -1, 84).permute(1, 2, 0), cmap='gray')
    # plt.show()
    # cv2.imshow('image', attack.to('cpu').reshape(1, -1, 84).permute(1, 2, 0).numpy())
    # cv2.waitKey(0)
    # exit()
    # return perturbed image
    return new_x, attack


def simba_mod(x, y, model, num_iters=500, epsilon=1):
    '''
    This is a modified version of SimBA where we only add positive values to the pixel brightess,
    setting a very high epsilon (max = 1), we basically perform a one/few-pixel attack
    '''

    num_pixels = x.reshape(1, -1).size(1)
    # get a random permutation of of the numbers from 0 to num_pixels-1
    perm = torch.randperm(num_pixels)

    # get the q_value from the model
    q_val = model(x.to('cuda'))[0, y].view(1,1)

    attack = torch.zeros(num_pixels)
    attack = attack.view(x.size())

    new_y = y
    i = 0

    # while the predictions are the same, keep trying
    while new_y == y and i < num_iters:

        delta = torch.zeros(num_pixels)
        delta[perm[i]] = epsilon
        delta = delta.view(x.size())

        # try adding epsilon * q = delta
        new_x = x + epsilon * delta

        # compute the q_value for x + delta +/- epsilon*q
        new_q_val = model(new_x.to('cuda')).max(1)[0].view(1,1)
        new_y = model(new_x.to('cuda')).max(1)[1].view(1,1)

        # if the updates q_value decreased
        if new_q_val < q_val:
            q_val = new_q_val
            x = new_x
            attack += (epsilon * delta)
        i += 1
    
    # print(i)
    # plt.imshow(new_x.to('cpu').reshape(1, -1, 84).permute(1, 2, 0), cmap='gray')
    # plt.show()
    # cv2.imshow('image', new_x.to('cpu').reshape(1, -1, 84).permute(1, 2, 0).numpy())
    # cv2.waitKey(0)
    # plt.imshow(attack.to('cpu').reshape(1, -1, 84).permute(1, 2, 0).numpy(), cmap='gray')
    # plt.show()
    # cv2.imshow('image', attack.to('cpu').reshape(1, -1, 84).permute(1, 2, 0).numpy())
    # cv2.waitKey(0)
    # exit()
    # return perturbed image
    return new_x, attack
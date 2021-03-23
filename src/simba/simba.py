'''
Code inspired from the official repository https://github.com/cg563/simple-blackbox-attack
'''

import torch

def simba(x, y, model, num_iters=50, epsilon=50):

    num_pixels = x.reshape(1, -1).size(1)
    # get a random permutation of of the numbers from 0 to num_pixels-1
    perm = torch.randperm(num_pixels)

    # get the q_value from the model
    q_val = model(x.to('cuda'))[0, y].view(1,1)

    new_y = y
    i = 0

    # while the predictions are the same, keep trying
    while new_y == y and i < num_iters:

        delta = torch.zeros(num_pixels)
        delta[perm[i]] = epsilon
        delta = delta.view(x.size())

        # try adding and subctracting epsilon * q = delta
        for e in {1, -1}:
            new_x = x + e * delta

            # compute the q_value for x + delta +/- epsilon*q#
            new_q_val = model(new_x.to('cuda')).max(1)[0].view(1,1)
            new_y = model(new_x.to('cuda')).max(1)[1].view(1,1)

            # if the updates q_value decreased
            if new_q_val < q_val:
                q_val = new_q_val
                x = new_x
                break
        i += 1
    
    # return perturbed image
    return new_x
# Copyright (c) EEEM071, University of Surrey

import torch


def init_lr_scheduler(
    optimizer,
    lr_scheduler="multi_step",  # learning rate scheduler
    stepsize=[20, 40],  # step size to decay learning rate
    gamma=0.1,  # learning rate decay
    T_max=20,
):
    if lr_scheduler == "single_step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=stepsize[0], gamma=gamma
        )

    elif lr_scheduler == "multi_step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=stepsize, gamma=gamma
        )
    elif lr_scheduler == "CosineAnnealingLR":        
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max , eta_min=0,last_epoch=-1
        )

    else:
        raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}")

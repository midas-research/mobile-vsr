# coding: utf-8
import os
import time
import random
import logging
import argparse
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torchsummary import summary


from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, TopKCategoricalAccuracy

from tqdm import tqdm

from utils import *
from model import *
from dataset import *
from lr_scheduler import *
from cvtransforms import *


print("Process Number: ",os.getpid())

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

use_gpu = True

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

torch.cuda.set_device(1)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def data_loader(args):
    dsets = {x: LRW(x, args.dataset) for x in ['train', 'val', 'test']}
    dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=args.batch_size,\
                       shuffle=True, num_workers=args.workers, pin_memory=use_gpu) \
                       for x in ['train', 'val', 'test']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val', 'test']}
    print('\nStatistics: train: {}, val: {}, test: {}'.format(dset_sizes['train'], dset_sizes['val'], dset_sizes['test']))
    # dset_loaders['train'] , dset_loaders['val']
    return dset_loaders, dset_sizes


def reload_model(model, path=""):
    if not bool(path):
        print('train from scratch')
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('*** model has been successfully loaded! ***')
        return model


def showLR(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def run(args, use_gpu=True):

    # saving
    save_path = os.path.join(os.getcwd(),'models')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    model = lipnext(inputDim=256, hiddenDim=512, nClasses=args.nClasses, frameLen=29, alpha=args.alpha)
    model = reload_model(model, args.path).to(device)

    dset_loaders, dset_sizes = data_loader(args)

    train_loader = dset_loaders['train']
    val_loader = dset_loaders['test']

    train_size = dset_sizes['train']
    val_size = dset_sizes['val']

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.)
    scheduler = AdjustLR(optimizer, [args.lr], sleep_epochs=5, half=5, verbose=1)
    # TQDM
    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=desc.format(0))

    # Ignite trainer
    trainer = create_supervised_trainer(model, optimizer, F.cross_entropy, \
                                        device=device, prepare_batch=prepare_train_batch)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 
                                                            'cross_entropy': Loss(F.cross_entropy),
                                                            'top-3': TopKCategoricalAccuracy(3)
                                                            }, device=device,\
                                                            prepare_batch=prepare_val_batch)

    # call backs
    @evaluator.on(Events.EPOCH_STARTED)
    def start_val(engine):
        tqdm.write(
            "Evaluation in progress"
        )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % args.interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(args.interval)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['cross_entropy']
        top_acc = metrics['top-3']
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f}, Top3: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, top_acc, avg_loss)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):

        # large dataset so saving often
        tqdm.write("saving model ..")
        torch.save(model.state_dict(), os.path.join(save_path,'epoch'+str(engine.state.epoch+1)+'.pt'))
        # saving to ONNX format
        dummy_input = torch.randn(args.batch_size, 1, 29, 88, 88)
        torch.onnx.export(model, dummy_input, "lipnext.onnx")

        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        top_acc = metrics['top-3']
        avg_loss = metrics['cross_entropy']
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f}, Top3: {:.2f} Avg loss: {:.2f} "
            .format(engine.state.epoch, avg_accuracy, top_acc, avg_loss)
        )
        

        pbar.n = pbar.last_print_n = 0

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_lr(engine):
        scheduler.step(engine.state.epoch)

    trainer.run(train_loader, max_epochs=args.epochs)
    pbar.close()

def main():
    # Settings
    args = parse_args()

    use_gpu = torch.cuda.is_available()
    run(args,use_gpu)

if __name__ == '__main__':
    main()

# main.py
import sys
import os
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import timm
from sklearn.model_selection import train_test_split
import uuid

from src.lr_schedulers import init_lr_scheduler
from src.optimizers import init_optimizer

from data import knifeDataset
from utils import *
from args import argument_parser, optimizer_kwargs, lr_scheduler_kwargs

# global variables
args = argument_parser().parse_args()

warnings.filterwarnings('ignore')

def train(train_loader, model, criterion, optimizer, scaler, scheduler, epoch, valid_accuracy, start, log):
    model.train()
    losses = AverageMeter()
    for i, (images, target, fnames) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = criterion(logits, target)
        
        losses.update(loss.item(), images.size(0))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        message_train = format_log_message("train", i, epoch, losses.avg, valid_accuracy[0], time_to_str(timer() - start, "sec"))
        print(f'\r{message_train}', end='', flush=True)
        
    message_train_epoch = format_log_message("train", i, epoch, losses.avg, valid_accuracy[0], time_to_str(timer() - start, "sec"))
    log.write("\n")
    log.write(message_train_epoch)

    return [losses.avg]

def evaluate(val_loader, model, criterion, epoch, train_loss, start, log):
    model.eval()
    map = AverageMeter()
    with torch.no_grad():
        for i, (images, target, fnames) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = model(images)
                preds = logits.softmax(1)
            
            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, target)
            map.update(valid_map5, images.size(0))
            # message = f'\rval  {i:5.1f} | {epoch:6.1f} | {train_loss[0]:.3f} | {map.avg:.3f} | {time_to_str(timer() - start, "min")}'
            message_val = format_log_message("val", i, epoch, train_loss[0], map.avg, time_to_str(timer() - start, "sec"))
            print(f'\r{message_val}', end='', flush=True)
        
        message_val_epoch = format_log_message("val", i, epoch, train_loss[0], map.avg, time_to_str(timer() - start, "sec"))
        log.write("\n")  
        log.write(message_val_epoch)
    
    return [map.avg]

def main():
    # Set the seed for reproducibility
    seed_value = args.seed
    set_seed(seed_value)
    
    ######################## load file and get splits #############################
    train_imlist = pd.read_csv(args.train_datacsv)
    train_gen = knifeDataset(train_imlist, mode="train")
    train_loader = DataLoader(train_gen, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    val_imlist = pd.read_csv(args.test_datacsv)
    val_gen = knifeDataset(val_imlist, mode="val")
    val_loader = DataLoader(val_gen, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ## Loading the model to run
    model = timm.create_model(args.model_mode, pretrained=True, num_classes=args.n_classes)
    model.to(device)
    
    ############################# Parameters #################################
    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))
    
    criterion = nn.CrossEntropyLoss().cuda()
    scaler = torch.cuda.amp.GradScaler()

    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
    log = Logger()
    # log.open("logs/%s_log_train.txt")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log.open(f"logs/log_train_{timestamp}.txt")
    student_id = os.environ.get('STUDENT_ID', 'your_id')
    student_name = os.environ.get('STUDENT_NAME', 'your_name')
    log.write(f"Student ID:{student_id}\n")
    log.write(f"Student name:{student_name}\n")
    log.write(f"UUID:{uuid.uuid4()}\n")

    log.write(f"==========\nArgs:{args}\n==========")

    log.write("\n" + "-" * 25 + f" [START {timestamp}] " + "-" * 25 + "\n\n")
    log.write('                           |  Train   |   Valid  |              |\n')
    log.write(' | Mode  |  Iter  | Epoch  |   Loss   |    mAP   |     Time     |\n')
    log.write('-' * 79 + '\n')

    # Execution control
    if args.evaluate_only:
        if args.model_path:
            model.load_state_dict(torch.load(args.model_path))
            model.to(device)
            print("Model loaded for evaluation.")
        else:
            raise ValueError("Model path must be specified for evaluation.")
        # Evaluate directly
        val_metrics = evaluate(val_loader, model, criterion, 0, [0], timer(), log)
        print("Evaluation complete.")
    else:
        start = timer()
        val_metrics = [0]
        
        for epoch in range(args.epochs):
            train_metrics = train(train_loader, model, criterion, optimizer, scaler, scheduler, epoch, val_metrics, start, log)
            val_metrics = evaluate(val_loader, model, criterion, epoch, train_metrics, start, log)

            filename = f"Knife-{args.model_mode}-E{epoch+1}.pth"
            if not os.path.exists(args.saved_checkpoint_path):
                os.mkdir(args.saved_checkpoint_path)
            save_path = os.path.join(args.saved_checkpoint_path, filename)
            torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()

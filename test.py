## import libraries for training
import warnings
from datetime import datetime
from timeit import default_timer as timer
import pandas as pd
import torch.optim
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import knifeDataset
import timm
from utils import *
warnings.filterwarnings('ignore')

# Validating the model
def evaluate(val_loader,model):
    model.cuda()
    model.eval()
    model.training=False
    map = AverageMeter()
    with torch.no_grad():
        for i, (images,target,fnames) in enumerate(val_loader):
            img = images.cuda(non_blocking=True)
            label = target.cuda(non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = model(img)
                preds = logits.softmax(1)
            
            valid_map5, valid_acc1, valid_acc5 = map_accuracy(preds, label)
            map.update(valid_map5,img.size(0))
    return map.avg

## Computing the mean average precision, accuracy 
def map_accuracy(probs, truth, k=5):
    with torch.no_grad():
        value, top = probs.topk(k, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))

        # top accuracy
        correct = correct.float().sum(0, keepdim=False)
        correct = correct / len(truth)

        accs = [correct[0], correct[0] + correct[1] + correct[2] + correct[3] + correct[4]]
        map5 = correct[0] / 1 + correct[1] / 2 + correct[2] / 3 + correct[3] / 4 + correct[4] / 5
        acc1 = accs[0]
        acc5 = accs[1]
        return map5, acc1, acc5

######################## load file and get splits #############################
print('reading test file')
test_files = pd.read_csv("test.csv")
print('Creating test dataloader')
test_gen = knifeDataset(test_files,mode="val")
test_loader = DataLoader(test_gen,batch_size=64,shuffle=False,pin_memory=True,num_workers=8)

print('loading trained model')
model = timm.create_model('tf_efficientnet_b0', pretrained=True,num_classes=config.n_classes)
model.load_state_dict(torch.load('Knife-Effb0-E9.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

############################# Training #################################
print('Evaluating trained model')
map = evaluate(test_loader,model)
print("mAP =",map)
    
   

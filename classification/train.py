import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import io
import random
from PIL import Image
import utils 
from tqdm import tqdm

def compute_adjustment(labels, tro=1.0):
    """compute the base probabilities"""
    N=len(labels)
    labels_unique=list(set(labels))
    print(labels_unique)
    label_freq_array = np.array([(np.array(labels)==label).sum()/N for label in labels_unique])
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    return adjustments

class RandomRotation90:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            i = random.randint(1, 3)

            if isinstance(x, Image.Image):
                x = transforms.RandomRotation((90*i, 90*i), expand=True)(x)

            elif isinstance(x, torch.Tensor):
                x = torch.rot90(x, i, [1, 2])

            else:
                raise TypeError(f'{type(x)} is unexpected type.')

        return x

def compress_image(img, quality_lower=80, quality_upper=100):
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=random.randint(quality_lower, quality_upper))
    buffer.seek(0)
    img = Image.open(buffer)
    return img



def crop_resize(img, size):
    w, h = img.size
    r = min(w, h)
    left = torch.randint(w - r + 1, size=(1,)).item()
    top = torch.randint(h - r + 1, size=(1,)).item()
    right = left + r
    bottom = top + r
    img = img.crop((left, top, right, bottom))
    img = img.resize((size, size), resample=Image.BILINEAR)
    return img

def log_confusion_matrix(writer, epoch, class_names, targets, predictions,phase):
    confusion = confusion_matrix(targets, predictions)
    fig, ax = plt.subplots()
    im = ax.imshow(confusion, cmap=plt.cm.Blues)
    ax.set_xticks(list(class_names.keys()))
    ax.set_yticks(list(class_names.keys()))
    ax.set_xticklabels(list(class_names.values()))
    ax.set_yticklabels(list(class_names.values()))
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(confusion.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(confusion.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=2)
    fig.colorbar(im)
    writer.add_figure(f'{phase}/Confusion Matrix', fig, epoch)

def training_step(model, criterion, optimizer, input, target,writer,global_step,class_names,adj):
    # move the input and target data to the GPU
    input = input.cuda()
    target = target.cuda()

   

    # forward pass
    output = model(input)
    loss = criterion(output+adj.to(output.device), target)

    # backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # calculate accuracy and log class-wise loss and accuracy to TensorBoard
    _, predicted = torch.max(output.data, 1)
    correct = (predicted == target).squeeze()
    for i in range(len(class_names)):
        class_correct = (predicted == target)[target == i].sum().item()
        class_total = (target == i).sum().item()
        if class_total > 0:
            writer.add_scalar(f'train/accuracy/{class_names[i]}', 100 * class_correct / class_total, global_step=global_step)
            writer.add_scalar(f'train/loss/{class_names[i]}', criterion(output[target == i], target[target == i]), global_step=global_step)
    writer.add_scalar('train/accuracy/total', 100 * correct.sum().item() / target.size(0), global_step=global_step)
    writer.add_scalar('train/loss/total', loss.item(), global_step=global_step)

    return 

def evaluate(model, criterion, val_loader, writer, epoch, class_names):
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    class_losses = torch.zeros(len(class_names)).cuda()
    class_correct = torch.zeros(len(class_names)).cuda()
    class_total = torch.zeros(len(class_names)).cuda()
    with torch.no_grad():
        for input, target in val_loader:
            # move the input and target data to the GPU
            input = input.cuda()
            target = target.cuda()
            # forward pass
            output = model(input)

            # calculate accuracy and predictions
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            # calculate per-class accuracy and loss
            for i in range(len(class_names)):
                class_mask = target == i
                class_correct[i] += (predicted[class_mask] == i).sum().item()
                class_total[i] += class_mask.sum().item()
                # class_loss = criterion(output[class_mask], target[class_mask])
                # class_losses[i] += class_loss.item()

    # calculate accuracy and log confusion matrix to TensorBoard
    accuracy = correct / total
    log_confusion_matrix(writer, epoch, class_names, all_targets, all_predictions,'val')
    writer.add_scalar('val/accuracy/total', accuracy, epoch)

    # log per-class accuracy and loss to TensorBoard
    for i in range(len(class_names)):
        if class_total[i] > 0:
            writer.add_scalar('val/accuracy/{}'.format(class_names[i]), class_correct[i] / class_total[i], epoch)
            # writer.add_scalar('val/loss/{}'.format(class_names[i]), class_losses[i] / class_total[i], epoch)

    return 

def main(args):
    # Please set the basedir 
	basedir=None
	assert basedir is not None, 'Please set the basedir'
	train_dir = os.path.join(basedir,"OpenAnimalTracks/classification/train")
	val_dir = os.path.join(basedir,"OpenAnimalTracks/classification/val")
	test_dir = os.path.join(basedir,"OpenAnimalTracks/classification/test")


    with open(args.config) as f:
        config = OmegaConf.load(f)

    from datetime import datetime
    now = datetime.now()
    formatted_date = now.strftime('%Y_%m_%d_%H_%M_%S')

    # set up TensorBoard logging
    log_dir = os.path.join('log', f'{config.model.name}{["","_linprob"][args.linprob]}/{formatted_date}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)


    batch_size = config.train.batch_size
    size = config.train.img_size
    train_transform = transforms.Compose([
        transforms.Resize((size,size)),
        RandomRotation90(p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Lambda(lambda img: compress_image(img, quality_lower=50, quality_upper=100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.train.mean, std=config.train.std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.train.mean, std=config.train.std)
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)


    adjustment = compute_adjustment(train_dataset.targets)
    if not args.use_adj:
        adjustment=torch.zeros_like(adjustment)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4,shuffle=True, drop_last=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False, pin_memory=True
    )

    class_names = {i: train_dataset.classes[i] for i in range(len(train_dataset.classes))}
    print(f'Class name: {class_names}')

    # define the model architecture
    model,head=utils.get_models(config.model.name,len(class_names),return_head=True)
    if args.weight is not None:
        pretrained_weight=torch.load(args.weight,map_location='cpu')
        model.load_state_dict(pretrained_weight)
        print(f'loaded from {args.weight}')
    model = model.to('cuda')
    model.eval()

    criterion = nn.CrossEntropyLoss()
    if args.linprob:
        params=head.parameters()
    else:
        params=model.parameters()
    optimizer = optim.SGD(params, lr=config.train.lr, momentum=config.train.momentum, weight_decay=config.train.weight_decay)
    
    global_step=0
    # train the model
    for epoch in range(config.train.num_epochs):
        if args.linprob:
            head.train()
        else:
            model.train()
        for i, (input, target) in enumerate(tqdm(train_loader,postfix=f"epoch={epoch+1}")):
            
            loss = training_step(model, criterion, optimizer, input, target,writer,global_step,class_names,adjustment)
            global_step+=1
        model.eval()
        accuracy = evaluate(model, criterion, val_loader, writer, epoch, class_names)


    weight_save_dir=f'checkpoints/{config.model.name}{["","_linprob"][args.linprob]}/{formatted_date}'
    os.makedirs(weight_save_dir, exist_ok=True)
    model_path=f'{weight_save_dir}/{config.train.num_epochs}.pth'
    torch.save(model.state_dict(), model_path)


if __name__=='__main__':

    # define command-line arguments
    parser = argparse.ArgumentParser(description='PyTorch image classification')
    parser.add_argument('-c', '--config', required=True, type=str, metavar='FILE', help='path to configuration file (default: config.yaml)')
    parser.add_argument('--use_adj',action='store_true')
    parser.add_argument('--linprob',action='store_true')
    parser.add_argument('--weight',default=None,type=str)
    args = parser.parse_args()
    main(args)

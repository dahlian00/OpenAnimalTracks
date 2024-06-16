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
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
from tqdm import tqdm

def plot_confusion_matrix(confusion,class_names,model_name):
	confusion=confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
	fig, ax = plt.subplots()
	im = ax.imshow(confusion, cmap=plt.cm.Blues,vmin=0, vmax=1)
	ax.set_xticks(list(class_names.keys()))
	ax.set_yticks(list(class_names.keys()))
	ax.set_xticklabels(list(class_names.values()))
	ax.set_yticklabels(list(class_names.values()))
	ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
	plt.setp(ax.get_xticklabels(), rotation=-60, ha="right", rotation_mode="anchor")
	for edge, spine in ax.spines.items():
		spine.set_visible(False)
	ax.set_xticks(np.arange(confusion.shape[1]+1)-.5, minor=True)
	ax.set_yticks(np.arange(confusion.shape[0]+1)-.5, minor=True)
	ax.grid(which="minor", color="gray", linestyle='-', linewidth=2)
	fig.colorbar(im)
	thresh = confusion.max() / 2.
	for i, j in itertools.product(range(confusion.shape[0]), range(confusion.shape[1])):
		if confusion[i, j]>0:
			plt.text(j, i, f'{confusion[i, j]:.02f}',fontsize= 'x-small',
					horizontalalignment="center",verticalalignment="center",
					color="white" if confusion[i, j] > thresh else "black")
	plt.subplots_adjust(left=0.1, right=1.0, bottom=0.025, top=0.825)
	plt.savefig(f'confmat_{model_name}.pdf',dpi=300)
	

def main(args):
	basedir=None
	assert basedir is not None, 'Please set the basedir'
	train_dir = os.path.join(basedir,"OpenAnimalTracks/classification/train")
	val_dir = os.path.join(basedir,"OpenAnimalTracks/classification/val")
	test_dir = os.path.join(basedir,"OpenAnimalTracks/classification/test")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	with open(args.config) as f:
		config = OmegaConf.load(f)

	size=config.train.img_size
	transform = transforms.Compose([
		transforms.Resize((size,size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=config.train.mean, std=config.train.std)
	])

	test_dataset = datasets.ImageFolder(test_dir, transform=transform)



	test_loader = torch.utils.data.DataLoader(
		test_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False, pin_memory=True
	)

	class_names = {i: test_dataset.classes[i] for i in range(len(test_dataset.classes))}
	# print(class_names)
	long2short_cvprw ={'alligator':'Alligator',
				'blackbear':'Bear',
				'beaver':'Beaver',
				'bobcat':'Cat',
				'coyote':'Coyote',
				'deer':'Deer',
				'duck':'Duck',
				'elephant':'Elephant',
				'redfox':'Fox',
				'horse':'Horse',
				'mountainlion':'Lion',
				'mouse':'Mouse',
				'nutria':'Nutria',
				'raccoon':'Raccoon',
				'hispidcottonrat':'Rat',
				'skunk':'Skunk',
				'squirrels':'Squirrel'}
	long2short_miru ={'alligator':'Alligator',
				'blackbear':'Bear',
				'beaver':'Beaver',
				'bobcat':'Cat',
				'coyote':'Coyote',
				'mule_deer':'Deer',
				'duck':'Duck',
				'elephant':'Elephant',
				'gray_fox':'Fox',
				'horse':'Horse',
				'mountainlion':'Lion',
				'mouse':'Mouse',
				'nutria':'Nutria',
				'raccoon':'Raccoon',
				'rat':'Rat',
				'western_spotted_skunk':'Skunk',
				'western_grey_squirrel':'Squirrel'}
	long2short ={'mink': 'Mink',
				'black_bear':'Bear',
				'beaver':'Beaver',
				'bob_cat':'Cat',
				'coyote':'Coyote',
				'muledeer':'Deer',
				'goose':'Goose',
				'elephant':'Elephant',
				'gray_fox':'Fox',
				'horse':'Horse',
				'lion':'Lion',
				'mouse':'Mouse',
				'turkey':'Turkey',
				'raccoon':'Raccoon',
				'rat':'Rat',
				'skunk':'Skunk',
				'western_grey_squirrel':'Squirrel',
				'otter':'Otter'}
	class_names_ordered=sorted([long2short[k] for k in long2short])
	short2idx = {class_names_ordered[i]:i for i in range(len(class_names_ordered))}
	# define the model architecture
	model_list=[]
	
	# model = models.__dict__[config.model.name](pretrained=config.model.pretrained)
	# model.fc = nn.Linear(model.fc.in_features, len(class_names))
	for i in range(args.n_models):
		model=utils.get_models(config.model.name,len(class_names))
		model.load_state_dict(torch.load(args.weight.replace('_0.pth',f'_{i}.pth'),map_location='cpu'))
		model = model.to('cuda')
		model.eval()
		model_list.append(model)

	
	y_true = []
	y_pred = [[] for _ in range(args.n_models)]
	with torch.no_grad():
		for inputs, labels in test_loader:
			

			inputs = inputs.to(device)
			labels = labels.to(device)
			for j in range(len(model_list)):
				outputs = model_list[j](inputs)
				_, preds = torch.max(outputs, 1)
				y_pred[j]+=preds.cpu().numpy().tolist()
			y_true+=labels.cpu().numpy().tolist()
				

	# calculate the accuracy and confusion matrix
	# acc = accuracy_score(y_true, y_pred)
	cm=0
	for i in range(args.n_models):
		a=confusion_matrix(y_true, y_pred[i])
		cm += a
	cm2=np.zeros_like(cm)
	for i in range(len(cm)):
		for j in range(len(cm[0])):
			cm2[i,j]=cm[short2idx[long2short[class_names[i]]],short2idx[long2short[class_names[j]]]]
	class_names_ordered={i:class_names_ordered[i] for i in range(len(class_names_ordered))}
	print(class_names_ordered)
	plot_confusion_matrix(cm2,class_names_ordered,args.config.split('/')[-1].split('.')[0])

	class_acc = []
	for i in range(len(class_names)):
		class_acc.append((cm2[i][i] / cm2[i].sum())*100)

	avg_class_acc = sum(class_acc) / len(class_names)

	text=f'{config.model.name} & '
	text+=''.join([f'{acc:0.2f} & ' for acc in class_acc])
	text+=f'{avg_class_acc:0.2f}'

	print(text)




if __name__=='__main__':

	# define command-line arguments
	parser = argparse.ArgumentParser(description='PyTorch image classification')
	parser.add_argument('-c', '--config', required=True, type=str, metavar='FILE', help='path to configuration file (default: config.yaml)')
	parser.add_argument('-w', '--weight', required=True, type=str, metavar='FILE', help='path to configuration file (default: config.yaml)')
	parser.add_argument('-n', '--n_models', required=True, type=int, metavar='FILE', help='path to configuration file (default: config.yaml)')
	args = parser.parse_args()
	main(args)

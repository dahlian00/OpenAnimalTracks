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
import torchvision
from torch.nn import functional as F


def denorm(x,mean,std):
	mean=torch.tensor(mean).reshape(1,3,1,1).to(x.device)
	std=torch.tensor(std).reshape(1,3,1,1).to(x.device)
	return x*std+mean

	
def get_attention(vit,x):
	attentions=[]
	x = vit._process_input(x)
	n = x.shape[0]

	# Expand the class token to the full batch
	batch_class_token = vit.class_token.expand(n, -1, -1)
	x = torch.cat([batch_class_token, x], dim=1)
	
	#Encoder
	# input = x
	x = x + vit.encoder.pos_embedding
	input = vit.encoder.dropout(x)
	for layer in vit.encoder.layers:
		x = layer.ln_1(input)
		x, attn = layer.self_attention(x, x, x, need_weights=True,average_attn_weights=True)
		attentions.append(attn)
		x = layer.dropout(x)
		x = x + input

		y = layer.ln_2(x)
		y = layer.mlp(y)
		input = x + y
	x = vit.encoder.ln(input)
	x = x[:, 0]
	x = vit.heads(x)
	# x = vit.encoder.ln(input)
	attentions = torch.stack(attentions, dim=1) # (n,num_layers,  num_heads, n_h * n_w, n_h * n_w)
	return x,attentions


	

def main(args):
	# Please set the basedir 
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
		test_dataset, batch_size=16, shuffle=False, num_workers=2, drop_last=False,
		pin_memory=True
	)

	class_names = {i: test_dataset.classes[i] for i in range(len(test_dataset.classes))}
	
	model=utils.get_models(config.model.name,len(class_names))
	model.load_state_dict(torch.load(args.weight,map_location='cpu'))
	model = model.to('cuda')
	model.eval()
	
	cnt_mat=np.zeros((len(class_names),len(class_names)))
	
	with torch.no_grad():
		for inputs, labels in test_loader:
			

			inputs = inputs.to(device)
			labels = labels.to(device)
			logits, attentions = get_attention(model,inputs)
			preds = logits.argmax(1)
			# print(attentions.shape,inputs.shape)
			
			class_token_attn_map = attentions[:,:, 0, 1:].mean(1)
			
			class_token_attn_map-=class_token_attn_map.min()
			class_token_attn_map/=class_token_attn_map.max()
			
			class_token_attn_map = class_token_attn_map.reshape(-1,7,7)
			cmap = plt.get_cmap('jet')
			class_token_attn_map = cmap(class_token_attn_map.cpu().numpy())  # (N, H, W, 4)
			# print(class_token_attn_map.shape)
			
			class_token_attn_map = torch.from_numpy(class_token_attn_map).to(inputs.device).permute(0, 3, 1, 2)[:, :3, :, :]  # (N, 3, H, W)
			inputs = denorm(inputs,config.train.mean, config.train.std)
			outputs = (F.interpolate(class_token_attn_map,inputs.shape[-2:],mode='bilinear') + inputs)/2
			for i in range(len(inputs)):
				label=labels[i].item()
				pred=preds[i].item()
				torchvision.utils.save_image(torch.cat([inputs[i:i+1],outputs[i:i+1]],0),f'output/{class_names[label]}_{class_names[pred]}_{cnt_mat[label,pred]}.png',normalize=True,value_range=(0,1),nrow=1)
				cnt_mat[label,pred]+=1
			
		




if __name__=='__main__':

	# define command-line arguments
	parser = argparse.ArgumentParser(description='PyTorch image classification')
	parser.add_argument('-c', '--config', required=True, type=str, metavar='FILE', help='path to configuration file (default: config.yaml)')
	parser.add_argument('-w', '--weight', required=True, type=str, metavar='FILE', help='path to configuration file (default: config.yaml)')
	args = parser.parse_args()
	main(args)

# best attack
# type "python hw5_best.py {input_folder} {output_folder}" to execute

import os
import sys
import time

import numpy as np
from PIL import Image

from torchvision.models import resnet50
from torchvision import transforms

import torch
from torch.autograd.gradcheck import zero_gradients
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax

# =================================
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean = mean,
								 std = std)
trans = transforms.Compose([transforms.ToTensor(), normalize])

nb_classes = 1000

if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')
# =================================

def load_data(input_folder):
	images = list()
	for i in range(200):
		img_name = str(i).rjust(3, '0') + '.png'
		img_path = os.path.join(input_folder, img_name)
		img = Image.open(img_path)
		img = np.asarray(img)
		images.append(img)

	labels = np.load('./label.npy')

	return images, labels

def deprocess(x_atk, x_ori, eps):
	x_atk = x_atk * np.array(std) + np.array(mean)
	x_atk = x_atk * 255
	diff = x_atk.astype('float64') - x_ori.astype('float64')
	diff[np.abs(diff) < 0.5 * eps] = 0
	sign = np.sign(diff)
	img = x_ori.astype('int64') + eps * sign
	img = np.clip(img, 0, 255).astype('uint8')

	return img

def fgsm(model, x_ori, label, eps = 1):
	atk_tensor = trans(np.copy(x_ori)).unsqueeze(0).to(device)
	atk_tensor.requires_grad = True
	y_pred = model(atk_tensor)
	y_true = torch.tensor(label, dtype = torch.long).unsqueeze(0).to(device)

	loss = CrossEntropyLoss()(y_pred, y_true)

	loss.backward()
	grad = atk_tensor.grad[0].cpu().numpy()
	grad = np.transpose(grad, (1, 2, 0))
	sign = np.sign(grad)
	x_atk = np.clip(x_ori.astype('int64') + eps * sign, 0, 255).astype('uint8')

	test_tensor = trans(np.copy(x_atk)).unsqueeze(0).to(device)
	pred_label = torch.argmax(model(test_tensor)[0]).item()

	return x_atk, (pred_label != label)

def iterative(model, x_ori, label, lr = 1, eps = 1, sign_attack = False):
	'''
	iterative gradient attack / iterative gradient sign attack
	'''
	repeat = 40000

	upper_arr = np.clip(x_ori.astype('int32') + eps, 0, 255).astype('uint8')
	lower_arr = np.clip(x_ori.astype('int32') - eps, 0, 255).astype('uint8')

	ori_tensor = trans(x_ori).unsqueeze(0).to(device)
	upper_arr = trans(upper_arr).unsqueeze(0).numpy() - 1e-6
	lower_arr = trans(lower_arr).unsqueeze(0).numpy() + 1e-6
	
	atk_arr = trans(np.copy(x_ori)).unsqueeze(0).numpy()

	predict = model(ori_tensor)
	ori_prob = softmax(predict[0])[label].item()

	y_true = torch.tensor(label, dtype = torch.long).unsqueeze(0).to(device)
	
	for i in range(repeat):
		atk_tensor = torch.from_numpy(atk_arr).to(device)
		atk_tensor.requires_grad = True
		y_pred = model(atk_tensor)
		loss = CrossEntropyLoss()(y_pred, y_true)

		loss.backward()
		grad = atk_tensor.grad
		atk_prob = softmax(y_pred[0])[label].item()

		if i % 500 == 0:
			print('iteration: %d | eps: %d | lr: %e | atk_prob: %.3f | ori_prob: %.3f' \
					% (i, eps, lr, atk_prob, ori_prob))
		if torch.argmax(y_pred).item() != label and atk_prob < 0.1:
			print('iteration: %d | eps: %d | lr: %e | atk_prob: %.3f | ori_prob: %.3f' \
					% (i, eps, lr, atk_prob, ori_prob))
			break

		r = grad[0].cpu().numpy()

		if sign_attack:
			atk_arr += (lr * np.sign(r) * 4e-5)
		else:
			atk_arr += (lr * r)

		# clip the arr
		atk_arr = np.clip(atk_arr, lower_arr, upper_arr)

	x_atk = np.transpose(atk_arr[0], (1, 2, 0))
	x_atk = deprocess(x_atk, x_ori, eps)

	test_tensor = trans(np.copy(x_atk)).unsqueeze(0).to(device)
	pred_label = torch.argmax(model(test_tensor)[0]).item()

	return x_atk, (pred_label != label)

def attack(model, x_train, y_train, output_folder):
	iter_custom_lr = {50: 100, 80: 100, 92: 1e7, 94: 10000, 142: 1e7, 182: 7000}

	for i in range(200):
		t = time.perf_counter()
		print('='*40)
		print('Attackimg image', i)
		x_ori = x_train[i]
		label = y_train[i]

		success = False
		if not success and i != 121:
			print('Using fgsm...')
			atk_arr, success = fgsm(model, x_ori, label)
			if success:
				print('fgsm successed!')
			else:
				print('fgsm failed!')

		if not success and i != 121:
			print('Using iterative...')
			if i in iter_custom_lr:
				atk_arr, success = iterative(model, x_ori, label, lr = iter_custom_lr[i])
			else:
				atk_arr, success = iterative(model, x_ori, label)

			if success:
				print('iterative successed!')
			else:
				print('iterative failed!')

		if i == 121:
			print('Encounter 121, it is very hard to attack, using sign attack with eps = 2...')
			atk_arr, success = iterative(model, x_ori, label, lr = 100, eps = 2, sign_attack = True)
			if success:
				print('successfully attacked picture 121!')
			else:
				print('falied to attack picture 121!')

		atk_img = Image.fromarray(atk_arr, mode = 'RGB')
		print('saving ' + str(i).rjust(3, '0') + '.png')
		atk_img.save(os.path.join(output_folder, str(i).rjust(3, '0') + '.png'))
		t = time.perf_counter() - t
		print('executing time: %.3f seconds' % t)

def main(script, input_folder, output_folder):
	x_train, y_train = load_data(input_folder)

	model = resnet50(pretrained = True)
	model.eval()
	model = model.to(device)

	attack(model, x_train, y_train, output_folder)

# ==================================
if __name__ == '__main__':
	t_all = time.perf_counter()
	main(*sys.argv)
	t_all = time.perf_counter() - t_all
	print('='*40)
	print('Total executing time: %.3f seconds' % t_all)
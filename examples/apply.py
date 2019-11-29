from __future__ import absolute_import, division, unicode_literals

import sys
import os
import torch
import logging
import numpy as np
from torch import nn

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
# get models.py from InferSent repo
from models import InferSent

# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'
# PATH_TO_W2V = 'glove/glove.840B.300d.txt'  # or crawl-300d-2M.vec for V2
PATH_TO_W2V = 'glove/glove.json'  # for small dataset
MODEL_PATH = 'infersent1.pkl'
V = 1 # version of InferSent

assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V), \
	'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval
from senteval import utils

def prepare(params, samples):
	params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False, is_small=True)


def batcher(params, batch):
	sentences = [' '.join(s) for s in batch]
	embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
	return embeddings

def convert_str2lst(s1):
	return [s.split() for s in s1]

"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
								 'tenacity': 3, 'epoch_size': 2}


def apply_logician(s1, s2 , is_list=False, sick_model = False):

	# is_list : If you are directly sending sentences then keep is_list = False
	#			If you are sending list of list of words then keep is_list = True

	# sick_model: if True, will use sick model for prediction
	#			: if False, will use snli model for prediction

	# Load InferSent model
	params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
					'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
	model = InferSent(params_model)
	model.load_state_dict(torch.load(MODEL_PATH))
	model.set_w2v_path(PATH_TO_W2V)

	params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
	params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
									 'tenacity': 3, 'epoch_size': 2}

	params_senteval['infersent'] = model.cpu()

	if not is_list:
		s1 = convert_str2lst(s1)
		s2 = convert_str2lst(s2)
	samples = s1+s2
	params_senteval['batch_size'] = min(128,len(s1))
	params_senteval = utils.dotdict(params_senteval)
	params_senteval.usepytorch  = True

	prepare(params_senteval, samples)

	emb_s1 = batcher(params_senteval, s1)
	emb_s2 = batcher(params_senteval, s2)
	if sick_model:
		testF = np.c_[ np.abs(emb_s1 - emb_s2),emb_s1 * emb_s2]
		cp = torch.load('./saved_sick.pth')
		print('[Contradiction  Neutral  Entailment]')
	else:
		testF = np.c_[emb_s1, emb_s2, emb_s1 * emb_s2, np.abs(emb_s1 - emb_s2)]
		cp = torch.load('./saved_snli_augment_ordered.pth', map_location=torch.device('cpu'))
		print('[ Entailment  Neutral Contradiction ]')
	inputdim = testF.shape[1]
	nclasses = 3
	clf = nn.Sequential(nn.Linear(inputdim, nclasses),).cpu()
	clf.load_state_dict(cp)

	testF = torch.FloatTensor(testF).cpu()
	out = clf(testF)
	sf = nn.Softmax(1)
	probs = sf(out)
	return probs


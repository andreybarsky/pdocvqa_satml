import argparse
import os, json, shutil
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

from build_utils import build_model
from utils import accuracy, anls

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import sklearn.cluster as cluster
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from scipy.stats import randint

SEEDS = [1026, 1027, 1028, 1029, 1030]
AZK_METRICS = [
	"accuracy", "anls", # G1
]
APK_METRICS = [
	"accuracy", "anls", # G1
	"confidence", "loss", # G2
	"delta_loss", "delta_confidence", # G3
]

LABEL_MEM = "member"
LABEL_NONMEM = "non_member"

def init_vt5(ckpt):
	visual_module_config = {'finetune': False, 'model': 'dit', 'model_weights': 'microsoft/dit-base-finetuned-rvlcdip'}
	experiment_config = {'model_weights': ckpt, 'max_input_tokens': 512, 'device': 'cuda', 'visual_module': visual_module_config}
	model = build_model('vt5', experiment_config)
	return model

def get_score_df(prov_train, prov_test, score, score_pt, dset):
	def create_df(provs):
		accuracy, anls, confidence, loss =  [], [], [], []
		delta_loss, delta_confidence, label = [], [], []
		for _p in provs:
			# G1
			accuracy.append(score[_p]['avg.accuracy']); anls.append(score[_p]['avg.anls'])

			# G2
			confidence.append(score[_p]['avg.confidence']); loss.append(score[_p]['avg.loss'])

			# G3
			delta_loss.append(
				np.abs(score[_p]['avg.loss'] - score_pt[_p]['avg.loss'])
			)
			delta_confidence.append(
				np.abs(score[_p]['avg.confidence'] - score_pt[_p]['avg.confidence'])
			)

			label.append(1 if score[_p]['label'] == LABEL_MEM else 0)

		return pd.DataFrame({
			'accuracy': accuracy, 'anls': anls,
			'confidence': confidence, 'loss': loss,
			'delta_loss': delta_loss, 'delta_confidence': delta_confidence,
			'label': label
		})

	test_df = create_df(prov_test)
	train_df = create_df(prov_train) if prov_train is not None else None
	return train_df, test_df

def run_inference(model, ckpt, data_dir, use_aux=False, k=100):
	if model == "vt5":

		model = init_vt5(ckpt)
		model.model.cuda()

		test_data = np.load(os.path.join(data_dir, "SaTML_PMIA_with_label.npy"), allow_pickle=True)
		if use_aux:
			aux_data = np.load(os.path.join(data_dir, "SaTML_PMIA_auxiliary.npy"), allow_pickle=True)[1:]
		vqa_data = np.concatenate((test_data, aux_data)) if use_aux else test_data

		score_dict = dict()
		for _ind,_record in tqdm(enumerate(vqa_data)):
			if _ind == 0: continue

			provider = _record['provider']
			document = _record['document']
			image = Image.open(os.path.join(data_dir, 'images' , _record['image_name'] + '.jpg')).convert("RGB")
			words = [word.lower() for word in _record['ocr_tokens']]
			boxes = np.array([bbox for bbox in _record['ocr_normalized_boxes']])
			question = _record['question']
			answer = _record['answers']

			_batch = [{
				'question_ids': _ind,
				'questions': question,
				'contexts': " ".join([_t.lower() for _t in _record['ocr_tokens']]),
				'answers': [answer.lower()],
				'image_names': _record['image_name'],
				'images': model.model.visual_embedding.feature_extractor(images=image, return_tensors="pt"),
				'words': words,
				'boxes': boxes,
			}]
			batch = {
				_k: [_d[_k] for _d in _batch] if _k != 'images'
				else {'pixel_values': torch.stack([_d[_k]['pixel_values'].squeeze(0) for _d in _batch])}
				for _k in _batch[0]
			}

			model.model.train()
			out, _, _ = model.forward(batch)

			model.model.eval()
			with torch.no_grad():
				_, preds, (logits, confidences) = model.forward(batch, return_pred_answer=True)

			topk = torch.topk(logits, k)
			logits, indices = topk.values.tolist(), topk.indices.tolist()

			if provider in score_dict:
				score_dict[provider]['accuracy'].append(accuracy(batch["answers"][0], preds[0]))
				score_dict[provider]['anls'].append(anls(batch["answers"][0], preds[0]))
				score_dict[provider]['loss'].append(out.loss.item())
				score_dict[provider]['confidence'].append(confidences[0])
				score_dict[provider]['prediction'].append(preds[0])
				score_dict[provider]['token_logit'].append(logits[0])
				score_dict[provider]['token_index'].append(indices[0])
				score_dict[provider]['answer'].append(batch["answers"][0])
				score_dict[provider]['document'].append(document)
				score_dict[provider]['index'].append(_ind if _ind < len(test_data) else (_ind-len(test_data)+1))
			else:
				score_dict[provider] = {
					'accuracy': [accuracy(batch["answers"][0], preds[0])],
					'anls': [anls(batch["answers"][0], preds[0])],
					'loss': [out.loss.item()],
					'confidence': [confidences[0]],
					'prediction': [preds[0]],
					'token_logit': [logits[0]],
					'token_index': [indices[0]],
					'answer': [batch["answers"][0]],
					'document': [document], 
					'index': [_ind if _ind < len(test_data) else (_ind-len(test_data)+1)],
					'label': _record['label'],
					'auxiliary': _ind > len(test_data)
				}
		for _p, _pdict in score_dict.items():
			assert len(_pdict['prediction']) == len(_pdict['accuracy']) == len(_pdict['anls']) \
					== len(_pdict['loss']) == len(_pdict['confidence']) \
					== len(_pdict['token_logit']) == len(_pdict['token_index'])
			_pdict.update({
				'avg.accuracy': np.mean(_pdict['accuracy']),
				'avg.anls': np.mean(_pdict['anls']),
				'avg.confidence': np.mean(_pdict['confidence']),
				'avg.loss': np.mean(_pdict['loss']),
			})
		acc_lst, anls_acc = [], []
		for _v in score_dict.values():
			acc_lst += _v['accuracy']; anls_acc += _v['anls']
		score_dict["overall"] = {"accuracy": np.mean(acc_lst), "anls": np.mean(anls_acc)}
		print(f"ACC={np.mean(acc_lst)}, ANLS={np.mean(anls_acc)}") 
	else:
		raise ValueError(f"Invalid model: {model}.")
	return score_dict

def run_kmeans(X, y, seed=None):
	kmeans = cluster.KMeans(init="random", n_clusters=2, n_init=10, max_iter=300, random_state=seed)
	kmeans.fit(X)
	
	acc_cluster_0 = np.mean([X[_ind][0] for _ind, _pred in enumerate(kmeans.labels_) if _pred == 0])
	acc_cluster_1 = np.mean([X[_ind][0] for _ind, _pred in enumerate(kmeans.labels_) if _pred == 1])

	y_pred = kmeans.labels_ if acc_cluster_0 <= acc_cluster_1 else (1 - kmeans.labels_)
	return accuracy_score(y, y_pred), balanced_accuracy_score(y, y_pred), f1_score(y, y_pred)

def run_random_forest(X_train, y_train, X_test, y_test, seed=None):
	param_dist = {'n_estimators': randint(50, 500), 'max_depth': randint(1, 20)}

	rf = RandomForestClassifier(random_state=seed)

	# Use random search to find the best hyperparameters
	rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5, random_state=seed)

	# Fit the random search object to the data
	rand_search.fit(X_train, y_train)
	best_rf = rand_search.best_estimator_

	y_preds = best_rf.predict(X_test)
	return accuracy_score(y_test, y_preds), balanced_accuracy_score(y_test, y_preds), f1_score(y_test, y_preds)

def run_unsupervised(args):
	data_dir = args.data_dir
	dset = data_dir.split('/')[-1]
	model = args.model
	score_dir = args.score_dir
	ckpt = args.ckpt
	pt_ckpt = args.pretrained_ckpt
	M = args.min_query

	if args.inference:
		assert ckpt != '', "Model checkpoint must be provided."
		score_dict_all = run_inference(model, ckpt, data_dir)
		with open(os.path.join(score_dir, f'{model}_{dset}_unsupervised.json'), 'w') as f:
			json.dump(score_dict_all, f)
	else:
		score_dict_all = json.load(open(os.path.join(score_dir, f'{model}_{dset}_unsupervised.json'), 'r'))
	if not os.path.exists(os.path.join(score_dir, f'{model}_{dset}_unsupervised_pt.json')):
		assert pt_ckpt != '', "Model checkpoint must be provided."
		score_dict_all_pt = run_inference(model, pt_ckpt, data_dir)
		with open(os.path.join(score_dir, f'{model}_{dset}_unsupervised_pt.json'), 'w') as f:
			json.dump(score_dict_all_pt, f)
	else:
		score_dict_all_pt = json.load(open(os.path.join(score_dir, f'{model}_{dset}_unsupervised_pt.json'), 'r'))
	score_dict = {_k:_v for _k,_v in score_dict_all.items() if _k != 'overall'}
	score_dict_pt = {_k:_v for _k,_v in score_dict_all_pt.items() if _k != 'overall'}
	assert set(score_dict.keys()) == set(score_dict_pt.keys())

	test_providers = list(score_dict.keys())

	avg_accs, bal_accs, f1_scores = [], [], []
	for _i, _seed in enumerate(SEEDS):
		_, score_df = get_score_df(None, test_providers, score_dict, score_dict_pt, dset)

		X, y = score_df[AZK_METRICS].values, score_df.label.values
		_avg_acc, _bal_acc, _f1 = run_kmeans(X, y, _seed)
		print(f"Seed={_seed}, AVG_ACC={_avg_acc*100:.2f}, BAL_ACC={_bal_acc*100:.2f}, F1={_f1*100:.2f}")
		avg_accs.append(_avg_acc); bal_accs.append(_bal_acc); f1_scores.append(_f1)

	avg_acc_mean = np.mean(np.array(avg_accs), axis=0)
	avg_acc_std = np.std(np.array(avg_accs), axis=0)
	print(f"Average Accuracy: {avg_acc_mean*100:.2f}±{avg_acc_std*100:.2f}")

	bal_acc_mean = np.mean(np.array(bal_accs), axis=0)
	bal_acc_std = np.std(np.array(bal_accs), axis=0)
	print(f"Balanced Accuracy: {bal_acc_mean*100:.2f}±{bal_acc_std*100:.2f}")

	f1_mean = np.mean(np.array(f1_scores), axis=0)
	f1_std = np.std(np.array(f1_scores), axis=0)
	print(f"F1 Score: {f1_mean*100:.2f}±{f1_std*100:.2f}")

def run_supervised(args):
	data_dir = args.data_dir
	dset = data_dir.split('/')[-1]
	model = args.model
	score_dir = args.score_dir
	ckpt = args.ckpt
	pt_ckpt = args.pretrained_ckpt
	M = args.min_query

	if args.inference:
		assert ckpt != '', "Model checkpoint must be provided."
		score_dict_all = run_inference(model, ckpt, data_dir, use_aux=True)
		with open(os.path.join(score_dir, f'{model}_{dset}_supervised.json'), 'w') as f:
			json.dump(score_dict_all, f)
	else:
		score_dict_all = json.load(open(os.path.join(score_dir, f'{model}_{dset}_supervised.json'), 'r'))
	if not os.path.exists(os.path.join(score_dir, f'{model}_{dset}_supervised_pt.json')):
		assert pt_ckpt != '', "Model checkpoint must be provided."
		score_dict_all_pt = run_inference(model, pt_ckpt, data_dir, use_aux=True)
		with open(os.path.join(score_dir, f'{model}_{dset}_supervised_pt.json'), 'w') as f:
			json.dump(score_dict_all_pt, f)
	else:
		score_dict_all_pt = json.load(open(os.path.join(score_dir, f'{model}_{dset}_supervised_pt.json'), 'r'))
	score_dict = {_k:_v for _k,_v in score_dict_all.items() if _k != 'overall'}
	score_dict_pt = {_k:_v for _k,_v in score_dict_all_pt.items() if _k != 'overall'}
	assert set(score_dict.keys()) == set(score_dict_pt.keys())

	train_providers = [_p for _p,_pdict in score_dict.items() if _pdict['auxiliary']]
	test_providers = [_p for _p,_pdict in score_dict.items() if not _pdict['auxiliary']]

	avg_accs, bal_accs, f1_scores = [], [], []
	for _i, _seed in enumerate(SEEDS):
		dftrain, dftest = get_score_df(train_providers, test_providers, score_dict, score_dict_pt, dset)
		Xtrain, ytrain = dftrain[APK_METRICS].values, dftrain.label.values
		Xtest, ytest = dftest[APK_METRICS].values, dftest.label.values
		_avg_acc, _bal_acc, _f1 = run_random_forest(Xtrain, ytrain, Xtest, ytest, _seed)
		print(f"Seed={_seed}, AVG_ACC={_avg_acc*100:.2f}, BAL_ACC={_bal_acc*100:.2f}, F1={_f1*100:.2f}")
		avg_accs.append(_avg_acc); bal_accs.append(_bal_acc); f1_scores.append(_f1)

	avg_acc_mean = np.mean(np.array(avg_accs), axis=0)
	avg_acc_std = np.std(np.array(avg_accs), axis=0)
	print(f"Average Accuracy: {avg_acc_mean*100:.2f}±{avg_acc_std*100:.2f}")

	bal_acc_mean = np.mean(np.array(bal_accs), axis=0)
	bal_acc_std = np.std(np.array(bal_accs), axis=0)
	print(f"Balanced Accuracy: {bal_acc_mean*100:.2f}±{bal_acc_std*100:.2f}")

	f1_mean = np.mean(np.array(f1_scores), axis=0)
	f1_std = np.std(np.array(f1_scores), axis=0)
	print(f"F1 Score: {f1_mean*100:.2f}±{f1_std*100:.2f}")

def parse_args():
	parser = argparse.ArgumentParser(description='script to run Provider MI attacks')

	parser.add_argument('--method', type=str, required=True, choices=['unsupervised','supervised'], help='attack method.')
	parser.add_argument('--data_dir', type=str, required=True, help='Provider MIA train/test data.')
	parser.add_argument('--model', type=str, required=True, help='model to attack.')
	parser.add_argument('--score_dir', type=str, required=True, help='pre-computed scores (.json) if any.')
	parser.add_argument('--inference', default=False, action='store_true', help='run inference to compute scores.')
	parser.add_argument('--ckpt', type=str, default='', help='model checkpoint to compute scores.')
	parser.add_argument('--pretrained_ckpt', type=str, default='', help='model pretrained checkpoint.')
	parser.add_argument('--min_query', type=int, default=0, help='minimum number of queries (lower bound).')

	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	print(args)

	if args.method == 'unsupervised':
		run_unsupervised(args)
	elif args.method == 'supervised':
		run_supervised(args)
	else:
		raise ValueError(f"Invalid method: {args.method}.")

if __name__ == '__main__':
	main()


import argparse
import os, json, shutil
import torch
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import accuracy, anls

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import sklearn.cluster as cluster
import sklearn.mixture as mixture
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_curve
from scipy.stats import randint, rankdata

SEEDS = [1026, 1027, 1028, 1029, 1030]
FEATURES = {
	"G1": ["accuracy", "anls"],
	"G2": ["confidence", "loss", "delta_loss", "delta_confidence"]
}

LABEL_MEM = "member"
LABEL_NONMEM = "non_member"

def seed_many(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

def init_vt5(ckpt):
	from models.VT5 import VT5
	visual_module_config = {'finetune': False, 'model': 'dit', 'model_weights': 'microsoft/dit-base-finetuned-rvlcdip'}
	experiment_config = {'model_name': 'vt5', 'model_weights': ckpt, 'max_input_tokens': 512, 'device': 'cuda', 'visual_module': visual_module_config}
	model = VT5(argparse.Namespace(**experiment_config))
	model.model.to(experiment_config['device'])
	return model

def get_score_df(prov_train, prov_test, score, score_pt, features):
	def create_df(provs):
		score_dict = dict({_k:[] for _k in features+['label']})
		for _p in provs:
			for _k in features:
				if 'delta' not in _k:
					score_dict[_k].append(score[_p][f'avg.{_k}'])
				else:
					score_dict[_k].append(
						np.abs(score[_p][f"avg.{_k.split('_')[-1]}"] - score_pt[_p][f"avg.{_k.split('_')[-1]}"])
					)
			score_dict['label'].append(score[_p]['label'])
		return pd.DataFrame(score_dict)

	test_df = create_df(prov_test)
	train_df = create_df(prov_train) if prov_train is not None else None
	return train_df, test_df

def run_inference(model, ckpt, suffix, data_dir, use_aux=False, k=100):
	if model == "vt5":
		model = init_vt5(ckpt)
		model.model.cuda()

		test_data = np.load(os.path.join(data_dir, f"SaTML_PMIA_test_{suffix}.npy"), allow_pickle=True)
		if use_aux:
			aux_data = np.load(os.path.join(data_dir, f"SaTML_PMIA_auxiliary_{suffix}.npy"), allow_pickle=True)[1:]
		vqa_data = np.concatenate((test_data, aux_data)) if use_aux else test_data
		label_dict = json.load(open(os.path.join(data_dir, f"SaTML_PMIA_label.json"), 'r'))[suffix]

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
				'images': image,
				'words': words,
				'boxes': boxes,
			}]
			batch = {_k: [_d[_k] for _d in _batch] for _k in _batch[0]}

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
				score_dict[provider]['index'].append(_ind if 'label' not in _record else (_ind-len(test_data)+1))
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
					'index': [_ind if 'label' not in _record else (_ind-len(test_data)+1)],
					'label': label_dict[provider],
					'auxiliary': 'label' in _record
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

def run_gaussian_mixture(X, y, seed=None, return_rank=False):
	gm = mixture.GaussianMixture(n_components=2, n_init=10, max_iter=300, random_state=seed)
	gm.fit(X)

	gm_pred = gm.predict(X)

	acc_cluster_0 = np.mean([X[_ind][0] for _ind, _pred in enumerate(gm_pred) if _pred == 0])
	acc_cluster_1 = np.mean([X[_ind][0] for _ind, _pred in enumerate(gm_pred) if _pred == 1])

	y_pred = gm_pred if acc_cluster_0 <= acc_cluster_1 else (1 - gm_pred)

	acc = accuracy_score(y, y_pred)
	bal_acc = balanced_accuracy_score(y, y_pred)
	f1 = f1_score(y, y_pred)

	probs = gm.predict_proba(X)[:,1] if acc_cluster_0 <= acc_cluster_1 else gm.predict_proba(X)[:,0]
	ranks = rankdata(probs, method='ordinal')
	fpr, tpr, _ = roc_curve(y, ranks)
	tpr_01fpr = tpr[np.where(fpr<.1)[0][-1]]
	tpr_003fpr = tpr[np.where(fpr<.03)[0][-1]]
	tpr_001fpr = tpr[np.where(fpr<.01)[0][-1]]
	if return_rank:
		return acc, bal_acc, f1, tpr_003fpr, ranks
	return acc, bal_acc, f1, tpr_01fpr, tpr_003fpr, tpr_001fpr

def run_kmeans(X, y, seed=None):
	kmeans = cluster.KMeans(init="random", n_clusters=2, n_init=10, max_iter=300, random_state=seed)
	kmeans.fit(X)
	
	acc_cluster_0 = np.mean([X[_ind][0] for _ind, _pred in enumerate(kmeans.labels_) if _pred == 0])
	acc_cluster_1 = np.mean([X[_ind][0] for _ind, _pred in enumerate(kmeans.labels_) if _pred == 1])

	y_pred = kmeans.labels_ if acc_cluster_0 <= acc_cluster_1 else (1 - kmeans.labels_)

	acc = accuracy_score(y, y_pred)
	bal_acc = balanced_accuracy_score(y, y_pred)
	f1 = f1_score(y, y_pred)

	probs = y_pred.astype(float)
	ranks = rankdata(probs, method='ordinal')
	fpr, tpr, _ = roc_curve(y, ranks)
	tpr_01fpr = tpr[np.where(fpr<.1)[0][-1]]
	tpr_001fpr = tpr[np.where(fpr<.01)[0][-1]]
	tpr_0001fpr = tpr[np.where(fpr<.001)[0][-1]]

	return acc, bal_acc, f1, tpr_01fpr, tpr_001fpr, tpr_0001fpr

def run_random_forest(X_train, y_train, X_test, y_test, seed=None, return_rank=False):
	param_dist = {'n_estimators': randint(50, 500), 'max_depth': randint(1, 20)}

	rf = RandomForestClassifier(random_state=seed)

	# Use random search to find the best hyperparameters
	rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5, random_state=seed)

	# Fit the random search object to the data
	rand_search.fit(X_train, y_train)
	best_rf = rand_search.best_estimator_

	y_pred = best_rf.predict(X_test)
	
	acc = accuracy_score(y_test, y_pred)
	bal_acc = balanced_accuracy_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)

	probs = best_rf.predict_proba(X_test)[:,1]
	ranks = rankdata(probs, method='ordinal')
	fpr, tpr, _ = roc_curve(y_test, ranks)
	tpr_01fpr = tpr[np.where(fpr<.1)[0][-1]]
	tpr_003fpr = tpr[np.where(fpr<.03)[0][-1]]
	tpr_001fpr = tpr[np.where(fpr<.01)[0][-1]]
	if return_rank:
		return acc, bal_acc, f1, tpr_003fpr, ranks
	return acc, bal_acc, f1, tpr_01fpr, tpr_003fpr, tpr_001fpr

def run_unsupervised(args):
	data_dir = args.data_dir
	dset = args.dataset
	model = args.model
	target = args.target
	score_dir = args.score_dir
	features = FEATURES["G1"]+FEATURES["G2"] if args.group == 'ALL' else FEATURES[args.group] 
	ckpt = args.ckpt
	pt_ckpt = args.pretrained_ckpt
	M = args.min_query

	if args.inference:
		assert ckpt != '', "Model checkpoint must be provided."
		score_dict_all = run_inference(model, ckpt, target, data_dir)
		with open(os.path.join(score_dir, f'{model}_{dset}_unsupervised.json'), 'w') as f:
			json.dump(score_dict_all, f)
	else:
		score_dict_all = json.load(open(os.path.join(score_dir, f'{model}_{dset}_unsupervised.json'), 'r'))
	if not os.path.exists(os.path.join(score_dir, f'{model}_{dset}_unsupervised_pt.json')):
		assert pt_ckpt != '', "Model checkpoint must be provided."
		score_dict_all_pt = run_inference(model, pt_ckpt, target, data_dir)
		with open(os.path.join(score_dir, f'{model}_{dset}_unsupervised_pt.json'), 'w') as f:
			json.dump(score_dict_all_pt, f)
	else:
		score_dict_all_pt = json.load(open(os.path.join(score_dir, f'{model}_{dset}_unsupervised_pt.json'), 'r'))
	score_dict = {_k:_v for _k,_v in score_dict_all.items() if _k != 'overall'}
	score_dict_pt = {_k:_v for _k,_v in score_dict_all_pt.items() if _k != 'overall'}
	assert set(score_dict.keys()) == set(score_dict_pt.keys())

	test_providers = list(score_dict.keys())

	avg_accs, bal_accs, f1_scores, tpr_01fprs, tpr_003fprs, tpr_001fprs = [], [], [], [], [], []
	for _i, _seed in enumerate(SEEDS):
		_, score_df = get_score_df(None, test_providers, score_dict, score_dict_pt, features)

		X, y = score_df[features].values, score_df.label.values
		_avg_acc, _bal_acc, _f1, _tpr_01fpr, _tpr_003fpr, _tpr_001fpr = run_gaussian_mixture(X, y, _seed)
		print(f"Seed={_seed}, AVG_ACC={_avg_acc*100:.2f}, BAL_ACC={_bal_acc*100:.2f}, F1={_f1*100:.2f}, " +\
			f"TPR@10%FPR={_tpr_01fpr*100:.2f}, TPR@3%FPR={_tpr_003fpr*100:.2f}, TPR@1%FPR={_tpr_001fpr*100:.2f}")
		avg_accs.append(_avg_acc); bal_accs.append(_bal_acc); f1_scores.append(_f1)
		tpr_01fprs.append(_tpr_01fpr); tpr_003fprs.append(_tpr_003fpr); tpr_001fprs.append(_tpr_001fpr)

	for _n, _lst in zip(
		["Accuracy", "Balanced Accuracy", "F1 Score", f"TPR@10%FPR", f"TPR@3%FPR", f"TPR@1%FPR"],
		[avg_accs, bal_accs, f1_scores, tpr_01fprs, tpr_003fprs, tpr_001fprs]
	):
		_mean = np.mean(np.array(_lst), axis=0)
		_std = np.std(np.array(_lst), axis=0)
		print(f"{_n}: {_mean*100:.2f}±{_std*100:.2f}")

def run_supervised(args):
	data_dir = args.data_dir
	dset = args.dataset
	model = args.model
	target = args.target
	score_dir = args.score_dir
	features = FEATURES["G1"]+FEATURES["G2"] if args.group == 'ALL' else FEATURES[args.group]
	ckpt = args.ckpt
	pt_ckpt = args.pretrained_ckpt
	M = args.min_query

	if args.inference:
		assert ckpt != '', "Model checkpoint must be provided."
		score_dict_all = run_inference(model, ckpt, target, data_dir, use_aux=True)
		with open(os.path.join(score_dir, f'{model}_{dset}_supervised.json'), 'w') as f:
			json.dump(score_dict_all, f)
	else:
		score_dict_all = json.load(open(os.path.join(score_dir, f'{model}_{dset}_supervised.json'), 'r'))
	if not os.path.exists(os.path.join(score_dir, f'{model}_{dset}_supervised_pt.json')):
		assert pt_ckpt != '', "Model checkpoint must be provided."
		score_dict_all_pt = run_inference(model, pt_ckpt, target, data_dir, use_aux=True)
		with open(os.path.join(score_dir, f'{model}_{dset}_supervised_pt.json'), 'w') as f:
			json.dump(score_dict_all_pt, f)
	else:
		score_dict_all_pt = json.load(open(os.path.join(score_dir, f'{model}_{dset}_supervised_pt.json'), 'r'))
	score_dict = {_k:_v for _k,_v in score_dict_all.items() if _k != 'overall'}
	score_dict_pt = {_k:_v for _k,_v in score_dict_all_pt.items() if _k != 'overall'}
	assert set(score_dict.keys()) == set(score_dict_pt.keys())

	train_providers = [_p for _p,_pdict in score_dict.items() if _pdict['auxiliary']]
	test_providers = [_p for _p,_pdict in score_dict.items() if not _pdict['auxiliary']]

	avg_accs, bal_accs, f1_scores, tpr_01fprs, tpr_003fprs, tpr_001fprs = [], [], [], [], [], []
	for _i, _seed in enumerate(SEEDS):
		dftrain, dftest = get_score_df(train_providers, test_providers, score_dict, score_dict_pt, features)
		Xtrain, ytrain = dftrain[features].values, dftrain.label.values
		Xtest, ytest = dftest[features].values, dftest.label.values
		_avg_acc, _bal_acc, _f1, _tpr_01fpr, _tpr_003fpr, _tpr_001fpr = run_random_forest(Xtrain, ytrain, Xtest, ytest, _seed)
		print(f"Seed={_seed}, AVG_ACC={_avg_acc*100:.2f}, BAL_ACC={_bal_acc*100:.2f}, F1={_f1*100:.2f}, " +\
			f"TPR@10%FPR={_tpr_01fpr*100:.2f}, TPR@3%FPR={_tpr_003fpr*100:.2f}, TPR@1%FPR={_tpr_001fpr*100:.2f}")
		avg_accs.append(_avg_acc); bal_accs.append(_bal_acc); f1_scores.append(_f1)
		tpr_01fprs.append(_tpr_01fpr); tpr_003fprs.append(_tpr_003fpr); tpr_001fprs.append(_tpr_001fpr)

	for _n, _lst in zip(
		["Accuracy", "Balanced Accuracy", "F1 Score", f"TPR@10%FPR", f"TPR@3%FPR", f"TPR@1%FPR"],
		[avg_accs, bal_accs, f1_scores, tpr_01fprs, tpr_003fprs, tpr_001fprs]
	):
		_mean = np.mean(np.array(_lst), axis=0)
		_std = np.std(np.array(_lst), axis=0)
		print(f"{_n}: {_mean*100:.2f}±{_std*100:.2f}")

def parse_args():
	parser = argparse.ArgumentParser(description='script to run Provider MIA baselines.')

	parser.add_argument('--method',required=True, choices=['unsupervised','supervised'], help='attack method.')
	parser.add_argument('--dataset', type=str, required=True, help='dataset.')
	parser.add_argument('--data_dir', type=str, required=True, help='Provider MIA train/test data.')
	parser.add_argument('--model', type=str, required=True, help='supported: VT5.')
	parser.add_argument('--target', required=True, choices=['dp','nondp'], help='model to attack.')
	parser.add_argument('--score_dir', type=str, required=True, help='pre-computed scores (.json) if any.')
	parser.add_argument('--group', type=str, required=True, choices=['G1','G2', 'ALL'], help='feature group.')
	parser.add_argument('--inference', default=False, action='store_true', help='run inference to compute scores.')
	parser.add_argument('--ckpt', type=str, default='', help='model checkpoint to compute scores.')
	parser.add_argument('--pretrained_ckpt', type=str, default='', help='model pretrained checkpoint.')
	parser.add_argument('--min_query', type=int, default=0, help='minimum number of queries (lower bound).')
	parser.add_argument('--seed', type=int, default=10, help='global seed.')

	args = parser.parse_args()
	return args

def main():
	args = parse_args()
	print(args)

	seed_many(args.seed)

	if args.method == 'unsupervised':
		run_unsupervised(args)
	elif args.method == 'supervised':
		run_supervised(args)
	else:
		raise ValueError(f"Invalid method: {args.method}.")

def some_stats():
	dp = json.load(open('vt5_satml_dp/vt5_satml_supervised.json' ,'r'))
	dp_acc, dp_anls, dp_mem_acc, dp_nonmem_acc, dp_mem_anls, dp_nonmem_anls = [], [], [], [], [], []
	for _k in dp:
		if _k != 'overall':
			dp_acc += dp[_k]['accuracy']
			if dp[_k]['label'] == 1: dp_mem_acc += dp[_k]['accuracy']
			if dp[_k]['label'] == 0: dp_nonmem_acc += dp[_k]['accuracy']
			dp_anls += dp[_k]['anls']
			if dp[_k]['label'] == 1: dp_mem_anls += dp[_k]['anls']
			if dp[_k]['label'] == 0: dp_nonmem_anls += dp[_k]['anls']
	print(f"DP(MEMBER/NONMEMBER) ACC={np.mean(dp_acc)}({np.mean(dp_mem_acc)}/{np.mean(dp_nonmem_acc)})")
	print(f"DP(MEMBER/NONMEMBER) ANLS={np.mean(dp_anls)}({np.mean(dp_mem_anls)}/{np.mean(dp_nonmem_anls)})")
	
	nondp = json.load(open('vt5_satml_nondp/vt5_satml_supervised.json' ,'r'))
	nondp_acc, nondp_anls, nondp_mem_acc, nondp_nonmem_acc, nondp_mem_anls, nondp_nonmem_anls = [], [], [], [], [], []
	for _k in nondp:
		if _k != 'overall':
			nondp_acc += nondp[_k]['accuracy']
			if nondp[_k]['label'] == 1: nondp_mem_acc += nondp[_k]['accuracy']
			if nondp[_k]['label'] == 0: nondp_nonmem_acc += nondp[_k]['accuracy']
			nondp_anls += nondp[_k]['anls']
			if nondp[_k]['label'] == 1: nondp_mem_anls += nondp[_k]['anls']
			if nondp[_k]['label'] == 0: nondp_nonmem_anls += nondp[_k]['anls']
	print(f"NONDP(MEMBER/NONMEMBER) ACC={np.mean(nondp_acc)}({np.mean(nondp_mem_acc)}/{np.mean(nondp_nonmem_acc)})")
	print(f"NONDP(MEMBER/NONMEMBER) ANLS={np.mean(nondp_anls)}({np.mean(nondp_mem_anls)}/{np.mean(nondp_nonmem_anls)})")

def sweep_multi(target='nondp'):
	from argparse import Namespace
	from utils import plot_roc_curve_multi

	args = Namespace(
		method='unsupervised', dataset='satml', data_dir='../mmMIA/data/satml', 
		model='vt5', target=target, score_dir=f'../pfldocvqa_eval/vt5_satml_{target}/', 
		group='ALL', inference=False, ckpt='', pretrained_ckpt='', min_query=0, seed=10
	)

	def load_score(model, dset, method, score_dir):
		score_dict_all = json.load(open(os.path.join(score_dir, f'{model}_{dset}_{method}.json'), 'r'))
		score_dict_all_pt = json.load(open(os.path.join(score_dir, f'{model}_{dset}_{method}_pt.json'), 'r'))
		score_dict = {_k:_v for _k,_v in score_dict_all.items() if _k != 'overall'}
		score_dict_pt = {_k:_v for _k,_v in score_dict_all_pt.items() if _k != 'overall'}
		return score_dict, score_dict_pt

	sup_dict, sup_pt_dict = load_score(args.model, args.dataset, 'supervised', args.score_dir)
	unsup_dict, unsup_pt_dict = load_score(args.model, args.dataset, 'unsupervised', args.score_dir)

	sup_train = [_p for _p,_pdict in sup_dict.items() if _pdict['auxiliary']]
	sup_test = [_p for _p,_pdict in sup_dict.items() if not _pdict['auxiliary']]
	unsup_train = [_p for _p,_pdict in unsup_dict.items() if _pdict['auxiliary']]
	unsup_test = [_p for _p,_pdict in unsup_dict.items() if not _pdict['auxiliary']]

	labels, scores, legends = [], [], []

	max_metric, max_seed, max_X, max_y = -np.inf, None, None, None
	for _i, _seed in enumerate(SEEDS):
		_, score_df = get_score_df(None, sup_test, sup_dict, sup_pt_dict, FEATURES["G1"])
		X, y = score_df[FEATURES["G1"]].values, score_df.label.values
		_, _, _, _tpr_003fpr, _rank = run_gaussian_mixture(X, y, _seed, return_rank=True)
		if _tpr_003fpr > max_metric:
			max_metric = _tpr_003fpr
			max_seed = _seed; max_X = _rank; max_y = y
	print(f"Unsupervised(G1): Seed={max_seed}, TPR@3%FPR={max_metric*100:.2f}")
	labels += [max_y]; scores += [max_X]; legends += ['Unsupervised(G1)']

	for _features, _legend in zip([FEATURES["G1"], FEATURES["G1"]+FEATURES["G2"]], ["G1", "ALL"]):
		max_metric, max_seed, max_X, max_y = -np.inf, None, None, None
		for _i, _seed in enumerate(SEEDS):
			dftrain, dftest = get_score_df(sup_train, sup_test, sup_dict, sup_pt_dict, _features)
			Xtrain, ytrain = dftrain[_features].values, dftrain.label.values
			Xtest, ytest = dftest[_features].values, dftest.label.values
			_, _, _, _tpr_003fpr, _rank = run_random_forest(Xtrain, ytrain, Xtest, ytest, _seed, return_rank=True)
			if _tpr_003fpr > max_metric:
				max_metric = _tpr_003fpr
				max_seed = _seed; max_X = _rank; max_y = y
		print(f"Supervised({_legend}): Seed={max_seed}, TPR@3%FPR={max_metric*100:.2f}")
		labels += [max_y]; scores += [max_X]; legends += [f'Supervised({_legend})']

	plot_roc_curve_multi(labels, scores, legends, f'SaTML {target.upper()} Model: Baselines', f'{args.dataset}_{target}')

if __name__ == '__main__':
	main()

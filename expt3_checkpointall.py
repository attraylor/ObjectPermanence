import os
import json
from baselines.inference_main import modified_inference_main

from plinko_imports.overlay_bbs_inference_folder import overlay_main
from plinko_imports.score_above_occluder import score_main
from pathlib import Path
import argparse
import torch.nn as nn
from baselines.models_factory import ModelsFactory
import sys
import subprocess
from collections import defaultdict


sys.path = ["/users/atraylor/anaconda/OP/lib/python3.7/site-packages"] + sys.path

def get_best_model_from_folder(fp, model_name="opnet"):
	folder_fp = os.path.join(fp, model_name)
	if "best_model.pth" in os.listdir(folder_fp):
		bmp = os.path.join(folder_fp, "best_model.pth")
	else:
		i_and_f = [(int(f.split("_")[0]), os.path.join(folder_fp, f)) for f in os.listdir(folder_fp) if f.split("_")[0].isdigit()]
		i_and_f.sort(reverse=True)
		bmp = i_and_f[0][1]
	return bmp

def test_main(args):
	splits = args.splits.split(",") #the different tests we will be running
	name = args.name #the name of the data (eg template_data_v10)
	batch_size = 2
	num_workers = 12
	device="cuda:0"
	model_name = "opnet"
	num_frames = 57
	total_scores = []
	total_results = []

	setting = args.setting

	folder_fp = os.path.join(args.checkpoints_path, model_name)
	i_and_f = [(int(f.split("_")[0]), os.path.join(folder_fp, f)) for f in os.listdir(folder_fp) if f.split("_")[0].isdigit()]
	i_and_f.sort()

	split_to_hist = defaultdict(list)
	for i, model_path in i_and_f:
		if i > args.max_epochs:
			continue
		model_name: str = "opnet"
		mcf: str = "configs/smaller_opnet.json"
		with open(mcf) as rf:
			model_config: str = json.load(rf)
		model: nn.Module = ModelsFactory.get_model(model_name, model_config, model_path)
		for spl in splits:
			results_dir = Path(args.checkpoints_path) / "interim_bbs"#"results/{}/{}/{}".format(name, setting, spl)
			data_head = os.path.join("data", name)
			inf_samples_dir = os.path.join(data_head, spl)
			inf_labels_dir = os.path.join(data_head, "{}_labels".format(spl))
			modified_inference_main(model, model_name, results_dir, inf_samples_dir, inf_labels_dir, 
										batch_size, num_workers, device, num_frames)
			#python ../plinko/src/overlay_bbs_inference_folder.py --name "${name}/${split}" --frames_dir "vid_out/frames/${name}/${setting}/${split}" --video_dir "vid_out/videos/${name}/${setting}/${split}" --model_name "${name}/${setting}/${split}/"
			score_args = {
							"name": name,
							"setting": setting,
							"split": spl,
							"results_dir": results_dir,
							}
			score_ns = argparse.Namespace(**score_args)
			#THERE ARE NO SUBDIRS FOR THIS ONE
			_, all_scores, results = score_main(score_ns)
			results["setting"] = setting
			results["split"] = spl
			total_scores += [all_scores]
			total_results.append(results)
			if args.save_videos == True:
				fd2 = Path(args.checkpoints_path) / "test_frames" / spl / i
				vd2 = Path(args.checkpoints_path) / "test_videos" / spl / i
				fd2.mkdir(parents=True, exist_ok=True)
				vd2.mkdir(parents=True, exist_ok=True)
				
				overlay_args = {
								"name": "{}/{}".format(name, spl),
								"frames_dir": fd2,
								"video_dir": vd2,
								"model_name": "{}/{}/{}/".format(name, setting, spl),
								"results_dir": results_dir,
								}
				overlay_ns = argparse.Namespace(**overlay_args)

				overlay_main(overlay_ns)
				subprocess.run("rm -f {}/*.png".format(fd2), shell=True)
		
	if not os.path.exists(args.results_dir):
		os.makedirs(args.results_dir)
	with open(os.path.join(args.results_dir, "{}_longterm_scores.json".format(setting)), "w+") as wf:
		json.dump(total_scores, wf, indent=4)
	with open(os.path.join(args.results_dir, "{}_longterm_results.json".format(setting)), "w+") as wf:
		json.dump(total_results, wf, indent=4)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='run tests over many different models')
	parser.add_argument("--checkpoints_path")
	parser.add_argument("--splits")
	parser.add_argument("--setting")
	parser.add_argument("--name")
	parser.add_argument("--results_dir")
	parser.add_argument("--max_epochs",type=int)
	parser.add_argument("--save_videos", action="store_true")
	args = parser.parse_args()
	test_main(args)
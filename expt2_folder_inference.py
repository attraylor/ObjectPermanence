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
	ecf = args.expt_config #Dictionary mapping setting : path to checkpoint.
	with open(ecf) as rf:
		expt_config = json.load(rf)
	splits = args.splits.split(",") #the different tests we will be running
	name = args.name #the name of the data (eg template_data_v10)
	batch_size = 2
	num_workers = 12
	device="cuda:0"
	model_name = "opnet"
	num_frames = 57
	for setting, checkpoints_path in expt_config.items():
		model_name: str = "opnet"
		model_path: str = get_best_model_from_folder(checkpoints_path)
		mcf: str = "configs/smaller_opnet.json"
		with open(mcf) as rf:
			model_config: str = json.open(rf)
		model: nn.Module = ModelsFactory.get_model(model_name, model_config, model_path)
		for spl in splits:
			fd2 = Path(checkpoints_path) / "test_frames" / spl
			vd2 = Path(checkpoints_path) / "test_videos" / spl
			fd2.mkdir(parents=True, exist_ok=True)
			vd2.mkdir(parents=True, exist_ok=True)
			results_dir = "results/{}/{}/{}".format(name, setting, spl)
			data_head = os.path.join("data", name)
			inf_samples_dir = os.path.join(data_head, spl)
			inf_labels_dir = os.path.join(data_head, "{}_labels".format(spl))
			modified_inference_main(model, model_name, results_dir, inf_samples_dir, inf_labels_dir, 
										batch_size, num_workers, device, num_frames)
			#python ../plinko/src/overlay_bbs_inference_folder.py --name "${name}/${split}" --frames_dir "vid_out/frames/${name}/${setting}/${split}" --video_dir "vid_out/videos/${name}/${setting}/${split}" --model_name "${name}/${setting}/${split}/"
			overlay_args = {
							"name": "{}/{}".format(name, spl),
							"frames_dir": fd2,
							"video_dir": vd2,
							"model_name": "{}/{}/{}/".format(name, setting, spl),
							}
			overlay_ns = argparse.Namespace(**overlay_args)

			overlay_main(overlay_ns)
			score_args = {
							"name": name,
							"setting": setting,
							"split": spl
							}
			score_ns = argparse.Namespace(**score_args)
			#THERE ARE NO SUBDIRS FOR THIS ONE
			score_main(score_ns)



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='run tests over many different models')
	parser.add_argument("--expt_config")
	parser.add_argument("--splits")
	parser.add_argument("--name")
	args = parser.parse_args()
	test_main(args)
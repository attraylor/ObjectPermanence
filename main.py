import json
import argparse
from typing import Dict, Any
import os
import sys
sys.path = ["/users/atraylor/anaconda/OP/lib/python3.7/site-packages"] + sys.path

from baselines.analyze_iou_offline import analyze_results
from baselines.training_main import training_main
from baselines.inference_main import trackers_inference_main, reasoning_inference_main
from baselines.cater_setup_inference import cater_setup_inference
from baselines.preprocess_perception_main import preprocess_main
from baselines.supported_models import INFERENCE_SUPPORTED_MODELS, TRAINING_SUPPORTED_MODELS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training and inference over the CATER data')
    subparsers = parser.add_subparsers()

    # create parser for the inference command
    inference_parser = subparsers.add_parser('inference')
    inference_parser.set_defaults(mode='inference')
    inference_parser.add_argument("--model_type", type=str, required=True, choices=INFERENCE_SUPPORTED_MODELS,
                                  help='name of model to run in experience')
    inference_parser.add_argument("--results_dir", type=str, required=True,
                                  help="a path to a a dictionary to save results videos and predictions output")
    inference_parser.add_argument("--inference_config", type=str, required=True,
                                  help="a path to config file for the experiment")
    inference_parser.add_argument("--model_config", type=str, required=False,
                                  help="a path to config file for the experiment")
    inference_parser.add_argument("--num_frames", type=int, required=True,
                                  help="number of frames for the inference sequence")

    # create parser for the inference command
    preprocess_parser = subparsers.add_parser('preprocess')
    preprocess_parser.set_defaults(mode='preprocess')
    preprocess_parser.add_argument("--results_dir", type=str, required=True,
                                  help="a path to a a dictionary to save results videos and predictions output")
    preprocess_parser.add_argument("--config", type=str, required=True,
                                  help="a path to config file for the experiment")

    # create parser for the training command
    training_parser = subparsers.add_parser('training')
    training_parser.set_defaults(mode="training")
    training_parser.add_argument("--model_type", type=str, required=True, choices=TRAINING_SUPPORTED_MODELS,
                                 help='name of model to run in experience')
    training_parser.add_argument("--model_config", type=str, required=True,
                                 help="a path to config file for the model hyper-parameters experiment")
    training_parser.add_argument("--training_config", type=str, required=True,
                                 help="a path to config file for the training experiment hyper-parameters")
    training_parser.add_argument("--num_frames", type=int, required=True,
                                 help="number of frames for the inference sequence")
    training_parser.add_argument("--prefixes", type=str, default="",
                                 help="valid prefixes in the training dataset to read as input. leave blank to read all prefixes")
    training_parser.add_argument("--name", type=str, required=True,
                                  help="name of the data ver")
    training_parser.add_argument("--setting", type=str, required=True,
                                  help="name of the data type (eg. id, sp)")
    training_parser.add_argument("--splits", type=str, required=True,
                                  help="test names, split on comma")
    training_parser.add_argument("--save_all", action="store_true",
                                  help="if true, save every model")
    training_parser.add_argument("--pretrained_model", type=str, default=None,
                                  help="pretrained model to load from")
    training_parser.add_argument("--taskid", type=str, default=None,
                                  help="taskid")

    # create a parser for offline results analysis
    analysis_parser = subparsers.add_parser('analysis')
    analysis_parser.set_defaults(mode='analysis')
    analysis_parser.add_argument("--predictions_dir", type=str, required=True, metavar='CATER/results',
                                 help='Path to a directory containing snitch predictions in json format')
    analysis_parser.add_argument("--labels_dir", type=str, required=True, metavar='CATER/labels',
                                 help='Path to a directory containing snitch location labels (GT labels) in json format')
    analysis_parser.add_argument("--containment_annotations", type=str, required=False, metavar='CATER/containment_annotations.txt',
                                 help='Path to a text file containing containment frames for each video in the dataset')
    analysis_parser.add_argument("--containment_only_static_annotations", type=str, required=False, metavar='CATER/containment_only_static_annotations.txt',
                                 help='Path to a text file containing only ststic (no movement) containment frames for each video in the dataset')
    analysis_parser.add_argument("--containment_with_movements_annotations", type=str, required=False, metavar='CATER/containment_with_move_annotations.txt',
                                 help='Path to a text file containing containment with movements frames for each video in the dataset')
    analysis_parser.add_argument("--visibility_ratio_gt_0", type=str, required=False, metavar='CATER/visibility_rate_gt_0.txt',
                                 help='Path to a text file containing annotations for frames where visiblity rate of the snithc is greater than 0%')
    analysis_parser.add_argument("--visibility_ratio_gt_30", type=str, required=False, metavar='CATER/visibility_rate_gt_30.txt',
                                 help='Path to a text file containing annotations for frames where visiblity rate of the snithc is greater than 30%')
    analysis_parser.add_argument("--visibility_ratio_gt_99", type=str, required=False, metavar='CATER/visibility_rate_gt_99.txt',
                                 help='Path to a text file containing annotations for frames where visiblity rate of the snithc is greater than 99%')
    analysis_parser.add_argument("--iou_thresholds", type=str, required=True, default="0.5,0.9",
                                 help='iou threshold for MAP calculation')
    analysis_parser.add_argument("--output_file", type=str, required=True, metavar="results.csv",
                                 help="Path to save the output csv file with the analyzed results")

    # create parser for the cater setup inference
    cater_parser = subparsers.add_parser('cater_inference')
    cater_parser.set_defaults(mode='cater_inference')
    cater_parser.add_argument("--results_dir", type=str, required=True,
                                  help="a path to a a dictionary to save classification results")
    cater_parser.add_argument("--inference_config", type=str, required=True,
                                  help="a path to config file for the experiment")
    cater_parser.add_argument("--model_config", type=str, required=False,
                                  help="a path to config file for the experiment")

    args = parser.parse_args()
    mode = args.mode
    if hasattr (args, "results_dir") and not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    if mode == "inference":
        model_type = args.model_type
        results_dir = args.results_dir
        inference_config_path = args.inference_config
        model_config_path = args.model_config

        # run experiment
        if model_type in TRAINING_SUPPORTED_MODELS:
            reasoning_inference_main(model_type, results_dir, inference_config_path, model_config_path,args.num_frames)

        else:
            trackers_inference_main(model_type, results_dir, inference_config_path)

    if mode == "preprocess":
        results_dir = args.results_dir
        config_path = args.config

        # run preprocess code
        preprocess_main(results_dir, config_path)

    if mode == "training":
        model_type = args.model_type
        model_config_path = args.model_config
        train_config_path = args.training_config
        
        # load model and training configuration for json files
        with open(model_config_path, "rb") as f:
            model_config: Dict[str, int] = json.load(f)

        with open(train_config_path, "rb") as f:
            train_config: Dict[str, Any] = json.load(f)
    
        if train_config.get("prefixes", "") == "":
            prefixes = []
        else:
            prefixes = train_config["prefixes"].split(",")
        print(prefixes)

        splits = args.splits.split(",")

        training_main(model_type, train_config, model_config, args.num_frames, 
					args.name, args.setting, splits, prefixes, args.save_all,
					args.pretrained_model, args.taskid)

    if mode == "analysis":
        predictions_dir = args.predictions_dir
        labels_dir = args.labels_dir
        output_file = args.output_file
        containment_annotations = args.containment_annotations
        containment_only_static = args.containment_only_static_annotations
        containment_with_move_annotations = args.containment_with_movements_annotations
        visibility_rate_gt_0 = args.visibility_ratio_gt_0
        visibility_rate_gt_30 = args.visibility_ratio_gt_30
        visibility_gt_99 = args.visibility_ratio_gt_99
        iou_threshold = [float(t) for t in args.iou_thresholds.split(",")]

        analyze_results(predictions_dir, labels_dir, output_file, containment_annotations, containment_only_static,
                        containment_with_move_annotations, visibility_rate_gt_0, visibility_rate_gt_30, visibility_gt_99,
                        iou_threshold)

    if mode == "cater_inference":
        model_type = "opnet"
        results_dir = args.results_dir
        inference_config_path = args.inference_config
        model_config_path = args.model_config

        cater_setup_inference(model_type, results_dir, inference_config_path, model_config_path)

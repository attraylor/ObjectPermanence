import os
import argparse
import json
from collections import defaultdict, Counter
import pickle
import numpy as np
import pandas as pd

def overlap(bb1, bb2):
	min1 = bb1["x1"]
	max1 = bb1["x2"]
	min2 = bb2["x1"]
	max2 = bb2["x2"]
	return max(0, min(max1, max2) - max(min1, min2))


def get_bb1_overlap(bb1, bb2):
    #BB1 is the target object-- how much of it is intersected with bb2?
    cond1 = bb1['x1'] < bb1['x2']
    cond2 = bb1['y1'] < bb1['y2']
    cond3 = bb2['x1'] < bb2['x2']
    cond4 = bb2['y1'] < bb2['y2']

    if not all([cond1, cond2, cond3, cond4]):
        print("cond is off!", bb1, bb2)
        return 0.0

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    return intersection_area * 1.0 / bb1_area


def get_ball_bounding_box(object):
	belief_x = object["location"][0]
	belief_y = object["location"][1]
	radius = object["radius"]
	model_pred_dict = {"x1": belief_x - radius, "x2": belief_x + radius, "y1": belief_y - radius, "y2": belief_y + radius}

	return model_pred_dict

def get_rect_bounding_box(object):
	belief_x = object["location"][0]
	belief_y = object["location"][1]
	width = object["width"]
	height = object["height"]
	model_pred_dict = {"x1": belief_x, "x2": belief_x + width, "y1": belief_y, "y2": belief_y + width}
	return model_pred_dict

def xywh_to_xyxy(box):
	x, y, w, h = box
	return [x, y, x+w, y+h]



def ball_in_correct_occluder(occluder1_bbs, occluder2_bbs, model_pred_dict, con_gt_dict):
	if occluder1_bbs is not None:
		occluder1_bb = xywh_to_xyxy(occluder1_bbs)
		occluder1_bb = to_dict(occluder1_bb)
		occluder1_iou = get_iou(model_pred_dict,occluder1_bb)
		gt_ball_in_occluder1 = overlap(con_gt_dict,occluder1_bb)
		occluder1_coverage = overlap(model_pred_dict, occluder1_bb)
	else: 
		gt_ball_in_occluder1 = 0
		occluder1_coverage = 0

	if occluder2_bbs is not None:
		occluder2_bb = xywh_to_xyxy(occluder2_bbs)
		occluder2_bb = to_dict(occluder2_bb)
		occluder2_iou = get_iou(model_pred_dict,occluder2_bb)
		gt_ball_in_occluder2 = overlap(con_gt_dict,occluder2_bb)
		occluder2_coverage = overlap(model_pred_dict, occluder2_bb)
	else: 
		gt_ball_in_occluder2 = 0
		occluder2_coverage = 0
	no_occluders = False
	if gt_ball_in_occluder1 == gt_ball_in_occluder2:
		no_occluders == True
	ball_in_occluder1 = False
	ball_in_occluder2 = False
	if gt_ball_in_occluder1 > gt_ball_in_occluder2:
		ball_in_occluder1 = True
	elif gt_ball_in_occluder1 < gt_ball_in_occluder2:
		ball_in_occluder2 = True


	if (ball_in_occluder1 and occluder1_coverage > occluder2_coverage) or \
	   (ball_in_occluder2 and occluder2_coverage > occluder1_coverage):
		occluderwide_acc = 1 #The ball is correctly inside the occluder.
	elif no_occluders == True and (occluder1_coverage < 0.5 or occluder2_coverage < 0.5):
		occluderwide_acc = 0 #The ball is incorrectly in a occluder.
	elif no_occluders == True and (occluder1_coverage < 0.5 and occluder2_coverage < 0.5):
		occluderwide_acc = 2 #Correctly, the ball is outside the occluder.
	elif occluder1_coverage > 0.5 or occluder2_coverage > 0.5:
		occluderwide_acc = 0 #The ball is in the incorrect occluder.
	else:
		occluderwide_acc = 2 #The ball is outside of a occluder.
	return occluderwide_acc

#https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    cond1 = bb1['x1'] < bb1['x2']
    cond2 = bb1['y1'] < bb1['y2']
    cond3 = bb2['x1'] < bb2['x2']
    cond4 = bb2['y1'] < bb2['y2']

    if not all([cond1, cond2, cond3, cond4]):
        return 0.0

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


CRITICAL_FRAMES = {
	"control_almostreal" : [31, 44],
	"control_hidden_in_cup" : [22, 38],
	"control_no_large" : [26, 40],
	"twocup" : [33, 46],
	"twocup_freebie" : [33, 46],
	"threecup" : [33, 46],
	"id_control": [41,46],
	"roll_behind_occluder_easy": [41,47],
	"roll_behind_occluder_hard": [41,47],
	"id_roll_viz": [57,63],
	"test": [41,46],
}

def to_dict(bb):
	assert len(bb) == 4
	return {"x1": bb[0], "x2": bb[2], "y1": bb[1], "y2": bb[3]}

def score_main(args):
	image_dir_path = os.path.join("data", args.name, args.split)
	gt_bbs_path = os.path.join("data", args.name , "{}_labels".format(args.split))

	bbs_path = os.path.join("results", args.name, args.setting, args.split)
	#bbs_path = os.path.join("results", args.model_name, taskid)

	write_dir = os.path.join("scores", args.name, args.setting, args.split)
	write_path = os.path.join(write_dir, "scores.json")
	weirds_file = os.path.join(write_dir, "weirds.json")

	if not os.path.exists(write_dir):
		os.makedirs(write_dir)


	total = 0
	after_correct = 0
	after_missing = 0

	before_correct = 0
	before_missing = 0

	accs = []
	score_pairs = []
	weirds = []
	for bbs_fp in os.listdir(bbs_path):
		code = bbs_fp.replace("_bb.json", "")
		config_type = "_".join(bbs_fp.split("_")[1:-2])
		subdir = os.path.join(image_dir_path, code)
		if not os.path.isdir(subdir):
			continue

		total += 1
		consistent_gt_bbs_fp = os.path.join(gt_bbs_path, code + "_bb.json")
		bbs_full_fp = os.path.join(bbs_path, bbs_fp)
		if not os.path.exists(bbs_full_fp):
			continue
		with open(bbs_full_fp) as rf:
			model_bbs = json.load(rf)

		with open(consistent_gt_bbs_fp) as rf:
			con_gt_bbs = json.load(rf)
			con_gt_ball_bbs = con_gt_bbs["ball1"]
			wedge_location = con_gt_bbs["wedge1"][0]
			wedge_midpoint = wedge_location[0] + 0.5 * wedge_location[2]
			occluder1_bbs = con_gt_bbs.get("occluder1", None)
			occluder2_bbs = con_gt_bbs.get("occluder2", None)
			if occluder1_bbs is not None:
				occluder1_bbs = occluder1_bbs[0]
			if occluder1_bbs is not None:
				occluder2_bbs = occluder2_bbs[0]


		crit_frames = CRITICAL_FRAMES[args.split]
		crit_correct = []
		crit_missing = []


		#before frame
		before_frame = crit_frames[0] - 1
		model_prediction = model_bbs[before_frame]
		consistent_gt = con_gt_ball_bbs[before_frame]
		x, y, w, h = consistent_gt
		con_gt_box = [x, y, x+w, y+h]
		model_pred_dict = {"x1": model_prediction[0], "x2": model_prediction[2], "y1": model_prediction[1], "y2": model_prediction[3]}
		con_gt_dict = {"x1": con_gt_box[0], "x2": con_gt_box[2], "y1": con_gt_box[1], "y2": con_gt_box[3]}

		con_iou = get_iou(model_pred_dict, con_gt_dict)

		predicted_ball_midpoint = model_prediction[0] + 0.5 * model_prediction[2]
		gt_ball_midpoint = consistent_gt[0] + 0.5 * consistent_gt[2]
		if (gt_ball_midpoint > wedge_midpoint and predicted_ball_midpoint > wedge_midpoint) or \
			(gt_ball_midpoint < wedge_midpoint and predicted_ball_midpoint < wedge_midpoint):
			correct = True
		else:
			correct = False
		
		score = ball_in_correct_occluder(occluder1_bbs, occluder2_bbs, model_pred_dict, con_gt_dict)
		correct_occluder = False
		missing = False

		if score == 1:
			before_correct += 1
			correct_occluder = True
		if score == 2:
			before_missing += 1
			missing = True
		#else it is wrong
		score_pairs_obj = {
				"before_or_after": "before",
				"con_dir_name": code,
				"config_type": config_type,
				"frame_num": before_frame,
				"con_iou": con_iou,
				"correct": correct,
				"correct_occluder": correct_occluder,
				"missing": missing,
				"split": args.split,
				"setting": args.setting,
				"score": score,
			}
		score_pairs.append(score_pairs_obj)


		#after frames
		mid_frame = int(np.ceil((crit_frames[0] + crit_frames[1] + 1)/2))
		#for critical_frame in range(crit_frames[0], crit_frames[1]+1):
		for critical_frame in [mid_frame]:
			model_prediction = model_bbs[critical_frame]
			consistent_gt = con_gt_ball_bbs[critical_frame]
			x, y, w, h = consistent_gt
			con_gt_box = [x, y, x+w, y+h]
			model_pred_dict = {"x1": model_prediction[0], "x2": model_prediction[2], "y1": model_prediction[1], "y2": model_prediction[3]}
			con_gt_dict = {"x1": con_gt_box[0], "x2": con_gt_box[2], "y1": con_gt_box[1], "y2": con_gt_box[3]}

			con_iou = get_iou(model_pred_dict, con_gt_dict)

			predicted_ball_midpoint = model_prediction[0] + 0.5 * model_prediction[2]
			gt_ball_midpoint = consistent_gt[0] + 0.5 * consistent_gt[2]
			if (gt_ball_midpoint > wedge_midpoint and predicted_ball_midpoint > wedge_midpoint) or \
			   (gt_ball_midpoint < wedge_midpoint and predicted_ball_midpoint < wedge_midpoint):
				correct = True
			else:
				correct = False
		
			score = ball_in_correct_occluder(occluder1_bbs, occluder2_bbs, model_pred_dict, con_gt_dict)
			correct_occluder = False
			missing = False

			if score == 1:
				correct_occluder = True
				after_correct += 1
			if score == 2:
				missing = True
				after_missing += 1

			crit_correct.append(correct_occluder)
			crit_missing.append(missing)



			#else it is wrong
			score_pairs_obj = {
				"before_or_after": "after",
				"con_dir_name": code,
				"config_type": config_type,
				"frame_num": critical_frame,
				"con_iou": con_iou,
				"correct": correct,
				"correct_occluder": correct_occluder,
				"missing": missing,
				"split": args.split,
				"setting": args.setting,
				"score": score,
			}
			score_pairs.append(score_pairs_obj)
		'''if all(crit_correct):
			#all labels are in agreement
			total_correct += 1
		elif any(crit_correct):
			weirds.append(code)
			#not all critical frames are on the same side of the wedge. potential prediction update!
			#print(code)'''
		
		accs.append(con_iou)

	acc_avg = sum(accs) * 1.0 / max(len(accs), 0.00001)
	#print("Saw {} pairs, avg iou {}.".format(total, acc_avg))
	with open(write_path, "w+") as wf:
		json.dump(score_pairs, wf, indent=4)
	with open(weirds_file, "w+") as wf:
		json.dump(weirds, wf, indent=4)
	
	b4c = before_correct * 1.0 / total
	b4m = before_missing * 1.0 / total
	b4i = (total - before_correct - before_missing) * 1.0 / total

	a4c = after_correct * 1.0 / total
	a4m = after_missing * 1.0 / total
	a4i = (total - after_correct - after_missing) * 1.0 / total

	c4c = a4c - b4c

	acc_dict = {
		"before_correct": b4c,
		"before_incorrect": b4i,
		"before_missing": b4m,
		"after_correct": a4c,
		"after_incorrect": a4i,
		"after_missing": a4m,
		"change": c4c
	}

	print(args.split, "{:.3f}\t{:.3f}\t{:.3f}  ||  {:.3f}\t{:.3f}\t{:.3f}  ||  {:.3f}".format(b4c, b4i, b4m, a4c, a4i, a4m, c4c))
	#print("BEFORE CORRECT", b4c, b4i, b4m, "AFTER_CORRECT", a4c, a4i, a4m)

	"""if args.split in ["control_almostreal", "threecup"]:
		with open(write_path) as rf:
			data = json.load(rf)
		df = pd.DataFrame(data)
		print(df.groupby(["before_or_after", "config_type"])["score"].value_counts() / df.groupby(["before_or_after", "config_type"])["score"].count())
	"""
	return acc_avg, score_pairs, acc_dict

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--name", type=str, default="template_data_v6")
	parser.add_argument("--scores_dir", type=str, default="scores")
	parser.add_argument("--setting", type=str, default="id")
	parser.add_argument("--split", type=str, default="twocup")
	parser.add_argument("--verbose", action="store_true")

	args = parser.parse_args()
	best_iou = {"iou": 0, "epoch_num": None}

	dirs = [d for d in os.listdir(args.scores_dir) if os.path.isdir(os.path.join(args.scores_dir, d))]
	num_epochs = len(dirs)

	score_main(args)


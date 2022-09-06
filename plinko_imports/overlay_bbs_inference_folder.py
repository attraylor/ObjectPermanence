import os
import argparse
os.environ['SDL_VIDEODRIVER'] = 'dummy'
import json

import pygame
import yaml
import pickle
import subprocess
bb_colors = {
140: (52, 189, 235, 255),
0: (52, 189, 235, 255),
1: (52, 189, 235, 255),
2: (52, 189, 235, 255),
3: (52, 189, 235, 255),
4: (52, 189, 235, 255),
5: (52, 189, 235, 255)
}
 
SIZE_FACTOR = 4.0


PPM = 20.0  # pixels per meter
TARGET_FPS = 5
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = int(640 / SIZE_FACTOR), int(480 / SIZE_FACTOR)
GROUND_HEIGHT = 1.0 / SIZE_FACTOR

MAX_X = 25 / SIZE_FACTOR


def make_frames(image_dir_path, gt_bbs_fp, bbs_path, d, out_dir):
	bbs_fp = os.path.join(bbs_path, d + "_bb.json")
	image_dir_fp = os.path.join(image_dir_path, d)
	#gt_bbs_fp = os.path.join(gt_bbs_path, d + "_bb.json")

	screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA, 32)
	pygame.display.set_caption('Simple pygame example')

	with open(bbs_fp) as rf:
		bbs = json.load(rf)
	with open(gt_bbs_fp) as rf:
		gt_bbs = json.load(rf)
		gt_ball_bbs = gt_bbs["ball1"]
		gt_wedge_bbs = gt_bbs["wedge1"]
		gt_cup1_bbs = gt_bbs.get("cup1", None)
		gt_cup2_bbs = gt_bbs.get("cup2", None)

	for i, (bb, gt_bb, gt_w_bb, gt_cup1_bb, gt_cup2_bb) in enumerate(zip(bbs, gt_ball_bbs, gt_wedge_bbs, gt_cup1_bbs, gt_cup2_bbs)):
		areas = []
		#load both images
		fp1 = os.path.join(image_dir_fp, "{0:04}.png".format(i))
		bg = pygame.image.load(fp1)
		screen.blit(bg, (0, 0))
		
		#areas.append((bottom_x - top_x) * (bottom_y - top_y))
		top_x, top_y, w, h = gt_cup1_bb
		pygame.draw.rect(screen, color=(255, 255, 255, 255), rect=[top_x, top_y, w, h])
		top_x, top_y, w, h = gt_cup2_bb
		pygame.draw.rect(screen, color=(180, 180, 180, 255), rect=[top_x, top_y, w, h])
		top_x, top_y, w, h = gt_bb
		pygame.draw.rect(screen, color=(151, 106, 253, 255), rect=[top_x, top_y, w, h])
		top_x, top_y, bottom_x, bottom_y = bb
		areas.append((bottom_x - top_x) * (bottom_y - top_y))
		pygame.draw.rect(screen, color=(52, 189, 235, 255), rect=[top_x, top_y, bottom_x - top_x, bottom_y - top_y])
		'''print(bbs_at_i)
		print(areas)
		a = input("q")'''
		pygame.image.save(screen, os.path.join(out_dir, "{0:04}.png".format(i)))
		screen.fill((0, 0, 0, 0))

def overlay_main(args):
	image_dir_path = os.path.join("data", args.name)
	gt_bbs_path = os.path.join("data", args.name + "_labels")
	bbs_path = os.path.join("results", args.model_name)


	for d in os.listdir(image_dir_path):
		subdir = os.path.join(image_dir_path, d)
		if not os.path.isdir(subdir):
			continue

		out_path = os.path.join(args.frames_dir, d)
		if not os.path.exists(out_path):
			os.makedirs(out_path)
		
		video_dir = os.path.join(args.video_dir)
		if not os.path.exists(video_dir):
			os.makedirs(video_dir)

		bbs_fp = os.path.join(bbs_path, d + "_bb.json")
		image_dir_fp = os.path.join(image_dir_path, d)
		gt_bbs_fp = os.path.join(gt_bbs_path, d + "_bb.json")

		screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA, 32)
		pygame.display.set_caption('Simple pygame example')

		with open(bbs_fp) as rf:
			bbs = json.load(rf)
		with open(gt_bbs_fp) as rf:
			gt_bbs = json.load(rf)
			gt_ball_bbs = gt_bbs["ball1"]
			gt_wedge_bbs = gt_bbs["wedge1"]

		for i, (bb, gt_bb, gt_w_bb) in enumerate(zip(bbs, gt_ball_bbs, gt_wedge_bbs)):
			areas = []
			#load both images
			fp1 = os.path.join(image_dir_fp, "{0:04}.png".format(i))
			bg = pygame.image.load(fp1)
			screen.blit(bg, (0, 0))
			top_x, top_y, w, h = gt_bb
			#areas.append((bottom_x - top_x) * (bottom_y - top_y))
			pygame.draw.rect(screen, color=(151, 106, 253, 255), rect=[top_x, top_y, w, h])
			top_x, top_y, bottom_x, bottom_y = bb
			areas.append((bottom_x - top_x) * (bottom_y - top_y))
			pygame.draw.rect(screen, color=(52, 189, 235, 255), rect=[top_x, top_y, bottom_x - top_x, bottom_y - top_y])
			pygame.image.save(screen, os.path.join(out_path, "{0:04}.png".format(i)))
			screen.fill((0, 0, 0, 0))
		video_path = os.path.join(video_dir, d + ".mp4")
		subprocess.run("ffmpeg -nostats -loglevel 0 -y -framerate 5 -pattern_type glob -i '{}/*.png' -c:v libx264 -pix_fmt yuv420p {}".format(out_path, video_path), shell=True)

		print("done")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--name", type=str)
	parser.add_argument("--model_name", type=str, default="worser_opnet")
	parser.add_argument("--frames_dir", type=str, default = "bb_comparison")
	parser.add_argument("--video_dir", type=str, default = "bb_comparison")

	args = parser.parse_args()
	overlay_main(args)
	
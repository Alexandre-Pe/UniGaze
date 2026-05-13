import os
import argparse
from datetime import datetime

import cv2
import face_alignment
import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.helper.image_transform import wrap_transforms
from gazelib.gaze.gaze_utils import pitchyaw_to_vector, vector_to_pitchyaw
from gazelib.gaze.normalize import estimateHeadPose, normalize
from gazelib.label_transform import get_face_center_by_nose
from utils import instantiate_from_cfg


def draw_gaze(image_in, pitchyaw, thickness=8, color=(0, 0, 255)):
	"""Draws a more 3D-like gaze vector on the image."""
	image_out = image_in.copy()
	(h, w) = image_in.shape[:2]
	length = w / 2.0
	pos = (int(h / 2.0), int(w / 2.0))

	if len(image_out.shape) == 2 or image_out.shape[2] == 1:
		image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

	dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
	dy = -length * np.sin(pitchyaw[0])
	end_point = (int(pos[0] + dx), int(pos[1] + dy))

	shadow_offset = 2
	shadow_color = (40, 40, 40)
	shadow_end = (end_point[0] + shadow_offset, end_point[1] + shadow_offset)
	cv2.arrowedLine(image_out, (pos[0] + shadow_offset, pos[1] + shadow_offset), shadow_end, shadow_color, thickness + 2, cv2.LINE_AA, tipLength=0.3)

	thickness_values = [4, 3, 2, 1]
	num_layers = len(thickness_values)
	for i in range(num_layers):
		alpha = i / num_layers
		layer_color = tuple(int((1 - alpha) * color[j] + alpha * 255) for j in range(3))
		cv2.arrowedLine(
			image_out,
			pos,
			end_point,
			layer_color,
			thickness_values[i],
			cv2.LINE_AA,
			tipLength=0.3,
		)

	return image_out


def denormalize_predicted_gaze(gaze_yaw_pitch, R_inv):
	pred_gaze_cancel_nor = pitchyaw_to_vector(gaze_yaw_pitch.reshape(1, 2)).reshape(3, 1)
	pred_gaze_cancel_nor = np.matmul(R_inv, pred_gaze_cancel_nor.reshape(3, 1))
	pred_gaze_cancel_nor = pred_gaze_cancel_nor / np.linalg.norm(pred_gaze_cancel_nor)
	pred_yaw_pitch_cancel_nor = vector_to_pitchyaw(pred_gaze_cancel_nor.reshape(1, 3))
	return pred_gaze_cancel_nor, pred_yaw_pitch_cancel_nor


def get_parser(**parser_kwargs):
	def str2bool(v):
		if isinstance(v, bool):
			return v
		if v.lower() in ("yes", "true", "t", "y", "1"):
			return True
		if v.lower() in ("no", "false", "f", "n", "0"):
			return False
		raise argparse.ArgumentTypeError("Boolean value expected.")

	parser = argparse.ArgumentParser(**parser_kwargs)

	parser.add_argument("--camera_id", default=0, type=int, help="OpenCV camera index (default: 0)")
	parser.add_argument("--cam_width", default=0, type=int, help="Camera capture width (0 means default)")
	parser.add_argument("--cam_height", default=0, type=int, help="Camera capture height (0 means default)")
	parser.add_argument("--resize_factor", default=0.5, type=float, help="Face detector resize factor")

	parser.add_argument("-out", "--output_dir", default=None, help="Path to save screenshots and optional recording")
	parser.add_argument("--window_name", default="UniGaze Live", type=str, help="OpenCV window name")
	parser.add_argument("--record", default=False, type=str2bool, help="Record rendered stream to mp4")
	parser.add_argument("--record_fps", default=30, type=int, help="Recording fps fallback when camera fps is unknown")
	parser.add_argument("--start_paused", default=False, type=str2bool, help="Start with the live stream paused")

	parser.add_argument("-m", "--model_cfg_path", help="Path to the model config file")
	parser.add_argument("--model_name", default=None, type=str, help="Load model directly from unigaze package")
	parser.add_argument("--ckpt_resume", default=None, type=str, help="Path to checkpoint")

	parser.add_argument(
		"--write_normalized_image",
		default=False,
		type=str2bool,
		help="Save normalized face crops with rendered gaze when faces are detected",
	)

	return parser


def set_dummy_camera_model(image=None):
	h, w = image.shape[:2]
	focal_length = w * 4
	center = (w // 2, h // 2)
	camera_matrix = np.array(
		[[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
		dtype="double",
	)
	camera_distortion = np.zeros((1, 5))
	return np.array(camera_matrix), np.array(camera_distortion)


def load_checkpoint(model, ckpt_key, ckpt_path):
	assert os.path.isfile(ckpt_path)
	weights = torch.load(ckpt_path, map_location="cpu")
	print("loaded ckpt from :", ckpt_path)

	model_state = weights[ckpt_key]
	if next(iter(model_state.keys())).startswith("module."):
		print("convert the DataParallel state to normal state")
		model_state = dict([(k[7:], v) for k, v in model_state.items()])

	model.load_state_dict(model_state, strict=True)
	print(f"loaded {ckpt_key}")
	del weights


def add_overlay_text(image, lines, color=(255, 255, 255)):
	font = cv2.FONT_HERSHEY_SIMPLEX
	scale = 0.55
	thickness = 1
	line_h = 22
	x = 12
	y = 26

	for i, line in enumerate(lines):
		y_i = y + i * line_h
		cv2.putText(image, line, (x + 1, y_i + 1), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
		cv2.putText(image, line, (x, y_i), font, scale, color, thickness, cv2.LINE_AA)


def process_frame(
	image_original,
	model,
	device,
	fa,
	image_torch_transform,
	face_model_load,
	facePts,
	resize_factor=0.5,
	focal_norm=960,
	distance_norm=600,
	roi_size=(224, 224),
	write_normalized_image=False,
	normalized_output_folder=None,
	frame_idx=0,
):
	arrow_colors = [(47, 255, 173)]
	output_image = image_original.copy()

	if resize_factor >= 1:
		image_resize = image_original.copy()
	else:
		image_resize = cv2.resize(
			image_original,
			dsize=None,
			fx=resize_factor,
			fy=resize_factor,
			interpolation=cv2.INTER_AREA,
		)

	image_resize = cv2.cvtColor(image_resize, cv2.COLOR_BGR2RGB)
	preds = fa.get_landmarks_from_image(image_resize)

	if preds is None:
		return output_image, 0

	num_faces = 0
	for idx in range(len(preds)):
		color = arrow_colors[idx % len(arrow_colors)]
		landmarks_in_original = preds[idx]
		landmarks_in_original /= resize_factor

		x_min = int(landmarks_in_original[:, 0].min())
		x_max = int(landmarks_in_original[:, 0].max())
		y_min = int(landmarks_in_original[:, 1].min())
		y_max = int(landmarks_in_original[:, 1].max())

		scale_factor_draw = 1.2
		bbox_width = x_max - x_min
		bbox_height = y_max - y_min
		bbox_center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
		x_min_draw = max(0, bbox_center[0] - int(bbox_width * scale_factor_draw // 2))
		x_max_draw = min(image_original.shape[1], bbox_center[0] + int(bbox_width * scale_factor_draw // 2))
		y_min_draw = max(0, bbox_center[1] - int(bbox_height * scale_factor_draw // 2))
		y_max_draw = min(image_original.shape[0], bbox_center[1] + int(bbox_height * scale_factor_draw // 2))

		scale_factor_crop = 2.0
		x_min = max(0, bbox_center[0] - int(bbox_width * scale_factor_crop // 2))
		x_max = min(image_original.shape[1], bbox_center[0] + int(bbox_width * scale_factor_crop // 2))
		y_min = max(0, bbox_center[1] - int(bbox_height * scale_factor_crop // 2))
		y_max = min(image_original.shape[0], bbox_center[1] + int(bbox_height * scale_factor_crop // 2))

		if x_max <= x_min or y_max <= y_min:
			continue

		image = image_original[y_min:y_max, x_min:x_max]
		landmarks = landmarks_in_original - np.array([x_min, y_min])

		camera_matrix, camera_distortion = set_dummy_camera_model(image=image)
		face_model = face_model_load[[20, 23, 26, 29, 15, 19], :]
		facePts_local = face_model.reshape(6, 1, 3) if facePts is None else facePts

		landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
		landmarks_sub = landmarks_sub.astype(float)
		landmarks_sub = landmarks_sub.reshape(6, 1, 2)
		hr, ht = estimateHeadPose(landmarks_sub, facePts_local, camera_matrix, camera_distortion)
		hR = cv2.Rodrigues(hr)[0]
		face_center_camera_cord, _ = get_face_center_by_nose(hR=hR, ht=ht, face_model_load=face_model_load)

		img_normalized, R, hR_norm, _, _, _ = normalize(
			image,
			landmarks,
			focal_norm,
			distance_norm,
			roi_size,
			face_center_camera_cord,
			hr,
			ht,
			camera_matrix,
			gc=None,
		)

		hr_norm = np.array([np.arcsin(hR_norm[1, 2]), np.arctan2(hR_norm[0, 2], hR_norm[2, 2])])
		if np.linalg.norm(hr_norm) > 80 * np.pi / 180:
			continue

		input_var = img_normalized[:, :, [2, 1, 0]]
		input_var = image_torch_transform(input_var)
		input_var = torch.autograd.Variable(input_var.float().to(device))
		input_var = input_var.unsqueeze(0)

		with torch.no_grad():
			ret = model(input_var)

		pred_gaze = ret["pred_gaze"][0]
		pred_gaze_np = pred_gaze.detach().cpu().numpy()
		num_faces += 1

		if write_normalized_image and normalized_output_folder is not None:
			img_normalized_drawn = draw_gaze(img_normalized, pred_gaze_np, thickness=5, color=color)
			cv2.imwrite(
				os.path.join(normalized_output_folder, f"frame_{frame_idx:06d}_face_{idx:02d}_normalized.jpg"),
				img_normalized_drawn,
			)

		R_inv = np.linalg.inv(R)
		pred_gaze_cancel_nor, _ = denormalize_predicted_gaze(pred_gaze_np, R_inv)
		vec_length = pred_gaze_cancel_nor * -112 * 1.5
		gazeRay = np.concatenate(
			(face_center_camera_cord.reshape(1, 3), (face_center_camera_cord + vec_length).reshape(1, 3)), axis=0
		)

		result = cv2.projectPoints(
			gazeRay,
			np.array([0, 0, 0]).reshape(3, 1).astype(float),
			np.array([0, 0, 0]).reshape(3, 1).astype(float),
			camera_matrix,
			camera_distortion,
		)
		result = result[0].reshape(2, 2)
		result += np.array([x_min, y_min])

		vector_start_point = (int(result[0][0]), int(result[0][1]))
		vector_end_point = (int(result[1][0]), int(result[1][1]))

		cv2.rectangle(output_image, (x_min_draw, y_min_draw), (x_max_draw, y_max_draw), (0, 0, 240), 2)

		shadow_offset = 2
		shadow_color = (40, 40, 40)
		shadow_end = (vector_end_point[0] + shadow_offset, vector_end_point[1] + shadow_offset)
		cv2.arrowedLine(
			output_image,
			(vector_start_point[0] + shadow_offset, vector_start_point[1] + shadow_offset),
			shadow_end,
			shadow_color,
			5,
			cv2.LINE_AA,
			tipLength=0.2,
		)

		thickness_values = [x * 3 for x in [4, 3, 2, 1]]
		num_layers = len(thickness_values)
		for i in range(num_layers):
			alpha = i / num_layers
			layer_color = tuple(int((1 - alpha) * color[j] + alpha * 255) for j in range(3))
			cv2.arrowedLine(
				output_image,
				vector_start_point,
				vector_end_point,
				layer_color,
				thickness_values[i],
				cv2.LINE_AA,
				tipLength=0.2,
			)

	return output_image, num_faces


if __name__ == "__main__":
	args, _ = get_parser().parse_known_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"

	if args.model_name is not None:
		import unigaze

		model = unigaze.load(args.model_name, device=device)  # type: ignore
		model.eval()
	else:
		pretrained_model_cfg = OmegaConf.load(args.model_cfg_path)["net_config"]  # type: ignore
		pretrained_model_cfg.params.custom_pretrained_path = None
		model = instantiate_from_cfg(pretrained_model_cfg)
		load_checkpoint(model, "model_state", args.ckpt_resume)
		model.eval()
		model.to(device)

	image_torch_transform = wrap_transforms("basic_imagenet", image_size=224)

	focal_norm = 960
	distance_norm = 600
	roi_size = (224, 224)

	face_model_load = np.loadtxt("data/face_model.txt")
	face_model = face_model_load[[20, 23, 26, 29, 15, 19], :]
	facePts = face_model.reshape(6, 1, 3)

	try:
		fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)  # type: ignore
	except Exception as e:
		print("Error initializing face_alignment:", e)
		raise e

	cap = cv2.VideoCapture(args.camera_id)
	if args.cam_width > 0:
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
	if args.cam_height > 0:
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)

	if not cap.isOpened():
		raise RuntimeError(f"Cannot open camera index {args.camera_id}")

	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS)
	if fps <= 1:
		fps = args.record_fps

	if args.output_dir is None:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		output_dir = os.path.join(os.getcwd(), f"camera_output_{timestamp}")
	else:
		output_dir = args.output_dir

	os.makedirs(output_dir, exist_ok=True)
	screenshot_dir = os.path.join(output_dir, "screenshots")
	os.makedirs(screenshot_dir, exist_ok=True)
	normalized_output_folder = os.path.join(output_dir, "normalized")
	if args.write_normalized_image:
		os.makedirs(normalized_output_folder, exist_ok=True)

	writer = None
	if args.record:
		fourcc = cv2.VideoWriter_fourcc(*"mp4v")
		record_path = os.path.join(output_dir, "camera_recording.mp4")
		writer = cv2.VideoWriter(record_path, fourcc, int(fps), (width, height))
		print(f"Recording enabled: {record_path}")

	print("============================ Controls ============================")
	print("s: start/resume stream")
	print("p or space: pause/resume stream")
	print("i: toggle gaze inference on/off")
	print("c: save screenshot")
	print("q or ESC: stop and quit")
	print("=================================================================")

	paused = args.start_paused
	inference_on = True
	frame_idx = 0
	screenshot_idx = 0
	last_frame = np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)
	window_name = args.window_name

	try:
		while True:
			if not paused:
				ret, frame = cap.read()
				if not ret:
					print("Camera read failed. Exiting.")
					break

				rendered = frame.copy()
				num_faces = 0
				if inference_on:
					rendered, num_faces = process_frame(
						image_original=frame,
						model=model,
						device=device,
						fa=fa,
						image_torch_transform=image_torch_transform,
						face_model_load=face_model_load,
						facePts=facePts,
						resize_factor=args.resize_factor,
						focal_norm=focal_norm,
						distance_norm=distance_norm,
						roi_size=roi_size,
						write_normalized_image=args.write_normalized_image,
						normalized_output_folder=normalized_output_folder,
						frame_idx=frame_idx,
					)

				status_lines = [
					f"Status: {'PAUSED' if paused else 'RUNNING'}",
					f"Inference: {'ON' if inference_on else 'OFF'} | Faces: {num_faces}",
					"Keys: s(start) p/space(pause) i(infer) c(capture) q/esc(quit)",
				]
				add_overlay_text(rendered, status_lines)

				last_frame = rendered
				frame_idx += 1
			else:
				rendered = last_frame.copy()
				status_lines = [
					"Status: PAUSED",
					f"Inference: {'ON' if inference_on else 'OFF'}",
					"Keys: s(start) p/space(pause) i(infer) c(capture) q/esc(quit)",
				]
				add_overlay_text(rendered, status_lines, color=(0, 220, 255))

			if writer is not None:
				writer.write(rendered)

			cv2.imshow(window_name, rendered)
			key = cv2.waitKey(1) & 0xFF

			if key in (ord("q"), 27):
				break
			if key in (ord("p"), ord(" ")):
				paused = not paused
				continue
			if key == ord("s"):
				paused = False
				continue
			if key == ord("i"):
				inference_on = not inference_on
				continue
			if key == ord("c"):
				screenshot_path = os.path.join(screenshot_dir, f"screenshot_{screenshot_idx:04d}.jpg")
				cv2.imwrite(screenshot_path, rendered)
				screenshot_idx += 1
				print(f"Saved screenshot: {screenshot_path}")

	finally:
		cap.release()
		if writer is not None:
			writer.release()
		cv2.destroyAllWindows()
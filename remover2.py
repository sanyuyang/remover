# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import time
import os


def get_alpha_W_white():
    video = "/users/yang/Downloads/white.mp4"
    cap = cv.VideoCapture(video)
    res, waterpic = cap.read()

    waterpic = waterpic.min(axis=-1)

    waterpic = waterpic.astype(np.float)
    waterpic[waterpic > 247] = 0
    waterpic[waterpic != 0] = 223.5

    alpha_0 = 32 / 255
    alpha = np.zeros(waterpic.shape, dtype=np.float)
    alpha[waterpic != 0] = alpha_0

    J = waterpic
    W = np.zeros(waterpic.shape, dtype=np.float)
    mask = waterpic != 0
    W[mask] = (J[mask] - (1 - alpha[mask]) * 255) / alpha[mask]

    alpha = np.repeat(alpha, 3, axis=-1).reshape(alpha.shape[0], alpha.shape[1], 3)
    W = np.repeat(W, 3, axis=-1).reshape(W.shape[0], W.shape[1], 3)
    return alpha, W


def get_alpha_W_black():
    video = '/users/yang/Downloads/black.mp4'

    cap = cv.VideoCapture(video)
    res, waterpic = cap.read()
    waterpic = waterpic.max(axis=-1)

    alpha_0 = 31.5 / 255
    thresh = 5

alpha = np.zeros(waterpic.shape, dtype=np.float)
alpha[waterpic > thresh] = alpha_0

J = waterpic
W = np.zeros(waterpic.shape, dtype=np.float)
mask = waterpic > thresh
W[mask] = (J[mask] - (1 - alpha[mask]) * 255) / alpha[mask]

alpha = np.repeat(alpha, 3, axis=-1).reshape(alpha.shape[0], alpha.shape[1], 3)
W = np.repeat(W, 3, axis=-1).reshape(W.shape[0], W.shape[1], 3)
return alpha, W
def merge_W(alpha, W):
bg = cv.imread("/users/yang/Downloads/bg.jpg")
bg = cv.resize(bg, (W.shape[1], W.shape[0]))
F = alpha * W + (1 - alpha) * bg
return F

def blur_mask(alpha):
mask = alpha[:, :, 0] > 0
mask = mask.astype(np.uint8) * 255
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
mask = cv.erode(mask, kernel, iterations=1)
mask = cv.medianBlur(mask, 9)
mask = cv.dilate(mask, kernel, iterations=1)
mask = mask.astype(np.float) / 255
mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
return mask

def find_offset(video):
cap = cv.VideoCapture(video)
res, frame = cap.read()
frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

for i in range(10):
    res, frame = cap.read()

frame_gray_next = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

flow = cv.calcOpticalFlowFarneback(frame_gray, frame_gray_next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

h, w = flow.shape[:2]
x, y = np.meshgrid(np.arange(w), np.arange(h))
fx, fy = flow[..., 0], flow[..., 1]

return int(round(np.mean(x + fx)) - x[0, 0]), int(round(np.mean(y + fy)) - y[0, 0])
def process_video(video_path, out_video_path):
alpha_white, W_white = get_alpha_W_white()
alpha_black, W_black = get_alpha_W_black()
alpha = alpha_white + alpha_black
W = W_white + W_black
F = merge_W(alpha, W)
mask = blur_mask(alpha)

cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)
frame_size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out_video = cv.VideoWriter(out_video_path, fourcc, fps, frame_size)

offset_x, offset_y = find_offset(video_path)

while True:
    res, frame = cap.read()
    if not res:
        break

    frame = cv.resize(frame, (W.shape[1], W.shape[0]))
    frame = cv.GaussianBlur(frame, (21, 21), 0)

    new_frame = np.zeros_like(frame)
    for i in range(3):
        new_frame[:, :, i] = cv.filter2D(frame[:, :, i], -1, mask[..., i])

    new_frame = new_frame.astype(np.uint8)

    out_frame = np.zeros_like(frame)
    out_frame[offset_y:, offset_x:, :] = new_frame[:-offset_y, :-offset_x, :]
    out_frame = out_frame.astype(np.uint8)

    out_video.write(out_frame)

cap.release()
out_video.release()
if name == "main":
video_path = "/users/yang/Downloads/test.mp4"
out_video_path = "/users/yang/Downloads/test_out.mp4"
process_video(video_path, out_video_path)

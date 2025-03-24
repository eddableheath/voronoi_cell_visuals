"""
Script for generating a video.
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from jsonargparse import CLI
from PIL import Image
from tqdm import tqdm

from vor_cell_vis.image_gen import gen_vor_image


def main(img_path: str):
    img = cv.imread(img_path)
    while True:
        cv.namedWindow("image", cv.WINDOW_AUTOSIZE)
        cv.imshow("image", img)
        if cv.waitKey(1) & 0xFF == ord("q"):
            cv.destroyAllWindows()
            break

    img = Image.fromarray(np.uint8(img)).convert("RGB")
    vor_img = gen_vor_image(img, 250, 0.8)
    vor_img.canvas.draw()
    vor_img_plot = np.array(vor_img.canvas.renderer.buffer_rgba())
    while True:
        cv.namedWindow("vor image", cv.WINDOW_AUTOSIZE)
        cv.imshow("vor image", cv.cvtColor(vor_img_plot, cv.COLOR_RGBA2RGB))
        if cv.waitKey(1) & 0xFF == ord("q"):
            cv.destroyAllWindows()
            break


def alt_main(vid_path: str):
    # load video and get details
    cap = cv.VideoCapture(vid_path)
    fps = int(cap.get(5))
    print("Frame Rate : ", fps, "frames per second")
    frame_count = cap.get(7)
    print("Frame count : ", frame_count)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_size = (frame_width, frame_height)
    print(frame_size)

    _, test_frame = cap.read()
    test_vor_plot = gen_vor_image(
        Image.fromarray(np.uint8(test_frame)).convert("RGB"), 50, 0.1
    )
    test_vor_plot.canvas.draw()
    test_vor_plot = np.array(test_vor_plot.canvas.renderer.buffer_rgba())
    vor_width = int(test_vor_plot.shape[0])
    vor_height = int(test_vor_plot.shape[1])
    vor_size = (vor_height, vor_width)
    print(vor_size)
    plt.close()

    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    out = cv.VideoWriter(
        "data/test_vids/anna_bday_out_3.mp4", fourcc, fps, vor_size, isColor=True
    )

    pbar = tqdm(total=frame_count)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        img = Image.fromarray(np.uint8(frame)).convert("RGB")
        vor_img = gen_vor_image(img, 50, 0.8)
        vor_img.canvas.draw()
        vor_img_plot = np.array(vor_img.canvas.renderer.buffer_rgba())
        out.write(cv.cvtColor(vor_img_plot, cv.COLOR_RGBA2RGB))
        plt.close()
        pbar.update(1)

        # cv.imshow("frame", cv.cvtColor(vor_img_plot, cv.COLOR_RGBA2RGB))
        # if cv.waitKey(1) & 0xFF == ord("q"):
        #     break

    cap.release()
    out.release()
    cv.destroyAllWindows()
    pbar.close()


if __name__ == "__main__":
    CLI(alt_main)

import ctypes
import glob
import os
import time
from multiprocessing import Queue, Process, Value
from queue import Empty

import cv2
import tqdm


def resize_image(q: Queue, outstanding_files: Value):
    while outstanding_files.value or q.qsize() > 0:
        try:
            image_name = q.get_nowait()
            dst_file = os.path.join(output_dir, image_name)
            fname = os.path.join(input_dir, image_name)
            image = cv2.imread(fname)
            image_resized = cv2.resize(image, out_size)
            cv2.imwrite(dst_file, image_resized)
        except Empty:
            time.sleep(0.1)


if __name__ == '__main__':
    input_dir = "/media/guy/Files 3/Tsofit/blindness detection/blindness2015/train"
    output_dir = "/media/guy/Files 3/Tsofit/blindness detection/train_images_2015_resized"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    out_size = (512, 512)
    files = []
    q = Queue(20)
    process_num = 8
    outstanding_files = Value(ctypes.c_bool, True)
    processes = [Process(target=resize_image, args=(q, outstanding_files)) for _ in range(process_num)]
    for p in processes:
        p.start()

    for image_name in tqdm.tqdm(glob.glob1(input_dir, '*.jpeg')):
        q.put(image_name)

    outstanding_files.value = False
    #     dst_file = os.path.join(output_dir, image_name)
    #     fname = os.path.join(input_dir, image_name)
    #     image = cv2.imread(fname)
    #     image_resized = cv2.resize(image, out_size)
    #     cv2.imwrite(dst_file, image_resized)
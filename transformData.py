"""
В данном файле производится преобразования данных
в спектрограммы единого размера.
"""
import pandas as pd
from itertools import count
import os 
import threading
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import csv
import itertools
import time
import matplotlib.pyplot as plt

fieldnames = ["rawEeg"]
drunk_folder = "./Drunk/"
sober_folder = "./Sober/"
new_drunk = "./data_drunk/"
new_sober = "./data_sober/"
drunk_images = "./drunk_images/"
sober_images = "./sober_images/"

segmentation = 4
#Искусственным образом преобразуем наши данные в 128Гц
Fs = 512 / segmentation
time_step = itertools.count(0)

def transform_txt_to_csv(class_name: str, folder: str, folder_csv: str) -> None:
    global time_step
    class_dir = os.listdir(folder)
    for count, item in enumerate(class_dir):
        with open(f"{folder_csv}{class_name}_{count}.csv", "w+") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

        with open(f"{folder}{item}", "r") as txt_file:
            logger.info(f"Transforming {item} into csv")
            with open(f"{folder_csv}{class_name}_{count}.csv", 'a') as csv_file:
                for i, rawEeg in enumerate(txt_file):
                    #Исключаем первые 3 секунды записи для калибровки
                    if i < 512 * 3:
                        continue
                    if i % segmentation != 0:
                        continue
                    #Записываем данные в csv файл
                    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    #Словарь с данными для записи
                    info = {
                        "rawEeg": int(rawEeg)
                    }
                    csv_writer.writerow(info)
            logger.info(f"{folder_csv}{class_name}_{count}.csv is created.")
            time_step = itertools.count(0)

def extract_spectr(folder: str, save_folder: str, num: int, duration: float) -> None:
    class_dir = os.listdir(folder)
    plt.figure(figsize=[1, 1])
    for count, item in enumerate(class_dir):
        logger.info(f"{folder}{item}")
        df = pd.read_csv(f"{folder}{item}")
        rawEeg = df["rawEeg"]
        Fs = 128
        n_fft = 128
        noverlap = 100
        for i in range(0, num):
            start = i * duration
            finish = start + duration
            plt.specgram(rawEeg[start:finish], noverlap=noverlap, NFFT=n_fft, Fs=Fs)
            plt.axis("off")
            logger.info(f"Segment:{i}")
            plt.tight_layout()
            plt.savefig(f'{save_folder}{item.replace(".csv", "")}_{i}.png', bbox_inches='tight', pad_inches=0)
            plt.clf()

if __name__ == "__main__":
    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     executor.submit(transform_txt_to_csv, "drunk", drunk_folder, new_drunk)
    #     executor.submit(transform_txt_to_csv, "sober", sober_folder, new_sober)
    extract_spectr(new_drunk, drunk_images, 5, 128 * 3)
    extract_spectr(new_sober, sober_images, 5, 128 * 3)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
import os
import cv2
import multiprocessing
import pickle 

def extract_spectr(file: str, num: int, duration: float, job: list = []) -> np.ndarray:
    dataset = []
    predict_folder = "./predict_data/"
    images_to_show = "./images_show/"
    df = pd.read_csv(file)
    rawEeg = df["rawEeg"]
    #Частота записи
    Fs = 128
    n_fft = 128
    noverlap = 100
    #Построение спектрограмм
    # fig, (axes_left, axes_right) = plt.subplots(nrows=2, ncols=2, figsize=[8, 8])
    # for i, ax in enumerate(zip(axes_left, axes_right)):
    #     start = i * duration * 2
    #     finish = start + duration
    #     data_to_show = (ax[0].specgram(rawEeg[start:finish], noverlap=noverlap, NFFT=n_fft, Fs=Fs),
    #                    ax[1].specgram(rawEeg[start + duration:finish + duration],
    #                    noverlap=noverlap, NFFT=n_fft, Fs=Fs))

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(data_to_show[0][3], cax=cbar_ax)
    # fig.savefig(f"{images_to_show}data_big_{i}.png", bbox_inches="tight", pad_inches=0)
        
    #Графики спектрограмм
    #Сохранение спектрограмм
    data_plot = []
    fig_save, ax_save = plt.subplots(figsize=[1, 1])
    for i in range(0, num):
        start = i * duration
        finish = start + duration
        data = ax_save.specgram(rawEeg[start:finish], noverlap=noverlap, NFFT=n_fft, Fs=Fs)
        ax_save.axis("off")
        fig_save.tight_layout()
        fig_save.savefig(f"{predict_folder}data_{i}.png", bbox_inches="tight", pad_inches=0)
        data_plot.append(data[0])
    pickle.dump(data_plot, open("data_plot", "wb"))
    
    #Создание датасета
    list_dir = os.listdir(predict_folder)
    for data in list_dir:
        image = cv2.imread(f"{predict_folder}{data}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.array(image).flatten()
        dataset.append(image)

    job.put(np.array(dataset))
    return np.array(dataset)
import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from data_extraction import extract_data
from spectr_extraction import extract_spectr
import threading
from loguru import logger
import sklearn
import pickle
import cv2
import numpy as np

file = "data.csv"
duration = 15
num_spec = 4
segmentation = 4
plt.style.use("seaborn")

#Объявление потоков со сбором данных и построением спектрограмм
data_thread = threading.Thread(target=extract_data, args=[file, segmentation, duration])
data_thread.start()

def animate(_) -> None:
    try:
        data = pd.read_csv(file)
        rawEeg = data["rawEeg"]
        point = np.arange(0, rawEeg.size, 1)/128
        #Обновление графика
        plt.cla()
        plt.plot(point, rawEeg, label="rawEeg")

    #Выход из программы по сочетанию Ctl+C, в связи
    #с работающими тредами.
    except KeyboardInterrupt:
        import sys
        sys.exit()

ani = FuncAnimation(plt.gcf(), animate, interval=100)
plt.tight_layout()
plt.show()

#Старт потока со спектрограмами
dataset = extract_spectr(file, num_spec, 128 * 3)

#Предксазываем класс полученных данных
model = pickle.load(open("svm_model.sav", 'rb'))
logger.info("Model is loaded")
prediction = model.predict(dataset)
logger.info(f"Result: {prediction}")
import socket 
import json
from loguru import logger
import time
from itertools import count
import csv

def extract_data(file: str = "data.csv", 
            segmentation: int = 5,
            extract_duration: int = 15, 
            jobs:list = []) -> None:
    #Сокет на котором ведется прослушивание сигналов
    HOST = '127.0.0.1'
    PORT = 13854
    #Json сигнал для гарнитуры отправлять raw
    buffer = json.dumps({"enableRawOutput": True, "format": "Json"})
    #Счетчик первых итераций [c]
    erase_time = 5

    #Поля в scv файле
    fieldnames = ["rawEeg"]
    time_step = count()
    point = 0

    #Открытие и очистка файла
    with open(file, "w") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    #Подключаемся к сокету типа TCP
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        #Общий цикл занимает в среднем 0.03 секунды, поэтому
        #надобности в time.sleep() нет, тк за такое время 
        #ресурсы CPU не пострадают.
        while True:
            try:
                s.send(buffer.encode("ascii"))
                data = s.recv(8192).decode("ascii")
                #Проверка состояния гарнитуры.
                if "status" in data:
                    #Начало времени ожидания колибровки
                    start_waiting = time.time()
                    continue
                #Первые 3 секунды обычно выдают не правильные значения,
                #поэтому было решено от них избавиться. Скорее всего отправляется
                #сохраненный буфер.
                if start_waiting + erase_time > time.time():
                    logger.info("Waiting for recording")
                    continue
                #Если rawEeg начали записываться, разделяем декодированное сообщение
                data = data.split("\r")
                #Исключение всех типов кроме "rawEeg"
                data = list(filter(lambda signal: "rawEeg" in signal, data))
                #Учитываем фактор не постоянного размера массива
                #Записываем данные в csv файл
                with open(file, 'a') as csv_file:
                    #Проводим выборку каждого segmentation единицы данных
                    for i in range(0, len(data), segmentation):
                        #Преобразование строк в словари
                        rawEeg = json.loads(data[i])["rawEeg"]
                        #Записываем данные в csv файл
                        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        #Словарь с данными для записи
                        point = next(time_step)
                        info = {
                            "rawEeg": rawEeg
                        }
                        csv_writer.writerow(info)
                #Остановка записи после определенного кол-ва времени
                if point > 128 * extract_duration:
                    break

            except KeyboardInterrupt:
                import sys
                sys.exit()
            
            except Exception as error:
                logger.info(error)

    jobs.append("Data extraction is done")
    logger.info("Data extraction is done")

# if __name__=="__main__":
    # extract_data()
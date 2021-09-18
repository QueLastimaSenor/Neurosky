import socket 
import json
import time
from itertools import count
import csv

from loguru import logger


class Neurosky:
    """
    Данный класс предназначен для подключения к гарнитуре
    Neurosky Mobile и извлечения данных RawEEG в csv файл.
    В данный момент совместим с Windows.

    ...
    Атрибуты
    --------
    self.HOST:str
        Адрес гарнитуры(const).
    self.PORT: int
        Порт гарнитуры(const).
    self.erase_time: int
        Время, которое должно быть пропущено со времени
        запуска записи. Данная атрибут необходим, тк 
        гарнитура первое время искажает данные.
    self.sample_rate: int
        Частота записи гарнитуры

    Методы
    ------
    extract_data() 
        Метод извлечения данных

        Аргументы
        ---------
        file: str
            Путь к файлу, в который будет проводиться запись
        segmentation: int
            Гарнитура может вести запись с частотой 256 или
            512 Гц, что не всегда необходимо для исследования,
            поэтому необходимо исскуственным методом понизить
            частоту записи.
        extract_duration: float
            Время записи сигнала
    """


    def __init__(self, erase_time:int=5):
        #Сокет на котором ведется прослушивание сигналов
        self.HOST: str = '127.0.0.1'
        self.PORT: int = 13854
        self.sample_rate: int = 512
        #Счетчик первых итераций [c]
        self.erase_time: int = erase_time

    def extract_data(self, file: str = "data.csv", 
                    segmentation: int = 5,
                    extract_duration: float = 15) -> None:

        #Json сигнал для гарнитуры отправлять raw
        buffer = json.dumps({"enableRawOutput": True, "format": "Json"})

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
            while True:
                try:
                    s.connect((self.HOST, self.PORT))
                    break

                except Exception as error:
                    logger.info(error)
                    time.sleep(1)
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
                    if start_waiting + self.erase_time > time.time():
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
                    if point > (self.sample_rate / segmentation) * extract_duration:
                        break

                except KeyboardInterrupt:
                    import sys
                    sys.exit()
                
                except Exception as error:
                    logger.info(error)

        logger.info("Data extraction is done")

if __name__ == "__main__":

    extractor = Neurosky(erase_time=5)
    extractor.extract_data()
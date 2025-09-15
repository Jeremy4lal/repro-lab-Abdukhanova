# repro-lab-Abdukhanova

Reproducibility lab — Python + PyTorch + Jupyter

Проект для лабораторной работы по теме реплицируемость экспериментов в PyTorch.
Содержит код для фиксации сида, обучения простой нейросети на синтетических данных и проверки воспроизводимости результатов.

Структура проекта repro-lab-Abdukhanova:
.
├── src/                 # исходный код
│   ├── seed_utils.py    # утилиты для фиксации случайных сидов
│   └── train.py         # основной скрипт обучения модели
│
├── notebooks/           # Jupyter Notebook с проверкой
│   └── 01-repro-seeds.ipynb
│
├── runs/                # сюда сохраняются модели и JSON-отчёты
│
├── requirements.txt     # список зависимостей
└── README.md            # документация проекта


Запуск:
1. python -m venv venv
2. source venv/bin/activate  # или venv\Scripts\activate (Windows)
3. pip install -r requirements.txt
4. pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
5. pre-commit install
6. python src/train.py --seed 13

Результаты сохраняются в `runs/`.

Установка/
1. Клонировать проект или скачать архив.
2. Создать виртуальное окружение: python -m venv venv
3. Активировать окружение: venv\Scripts\activate (для Windows) или source venv/bin/activate (для Linux/Mac)

Запуск обучения^
Пример запуска обучения модели: python src/train.py --seed 13 --epochs 3 --batch_size 32 --lr 0.05

После выполнения в папке runs/ появятся:
1. сохранённая модель: model_seed13.pt
2. отчёт о запуске: run_seed13.json

Проверка воспроизводимости^
В папке notebooks/ есть ноутбук 01-repro-seeds.ipynb.
Нужно открыть его в Jupyter Notebook или VS Code и запустить ячейки:
!python ../src/train.py --seed 13 --epochs 3 --batch_size 32 --lr 0.05
!python ../src/train.py --seed 13 --epochs 3 --batch_size 32 --lr 0.05

Результаты (final_loss, model_sha256) должны совпадать.

Аргументы скрипта train.py^
| Аргумент       | Тип   | Значение по умолчанию | Описание                  |
| -------------- | ----- | --------------------- | ------------------------- |
| `--seed`       | int   | 13                    | Сид для воспроизводимости |
| `--batch_size` | int   | 32                    | Размер батча              |
| `--epochs`     | int   | 3                     | Количество эпох обучения  |
| `--lr`         | float | 0.05                  | Скорость обучения         |

Версии библиотек (точные версии фиксируются в requirements.txt):
Python 3.10+
PyTorch >=2.0
NumPy >=1.24

Итог должен быть к примеру таким:
Training finished. Final loss: 0.08933366
Model saved to runs/model_seed13.pt, SHA256: 489670619fe7302879388f57e48adc17dd1458a6448d344d56ae4b526b440d61

Данные показатели говорят о том, что:
1. Модель успешно обучилась на синтетических данных (3 эпохи, batch=32, lr=0.05).
Final loss: 0.08933366 — итоговое значение функции потерь после обучения.
2. Файл модели сохранён в папке runs/ под названием model_seed13.pt.
Он содержит веса обученной нейросети.
3. SHA256-хэш модели рассчитан.
48967061... — это криптографическая "подпись" файла модели.
Если запустить обучение с теми же параметрами ещё раз, то финальный loss и SHA256 должны совпасть → это и будет доказательством воспроизводимости.

Если также отображается, это значит:
1. фиксация сидов работает,
2. DataLoader детерминированный,
3. окружение правильно настроено (CPU-only),
4. код соответствует требованиям задания.
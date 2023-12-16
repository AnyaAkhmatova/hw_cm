# CM project

Проект - домашнее задание Voice Anti-Spoofing по курсу dla. Предназначен для обучения countermeasure system (CM). В данном проекте была имплементирована и обучена модель S1-RawNet2. EER(eval) финальной модели = 0.0469 < 0.058.

____

За основу репозитория взят темплейт https://github.com/WrathOfGrapes/asr_project_template.git. Структура проекта изменена, большинство базовых классов были удалены, некоторые - перенесены в директории с классами-наследниками. Некоторые модули были удалены. Немного изменилась структура конфигов. В целом, просто адаптировали проект под реализацию RawNet2. 

Код для обучения модели - в hw_cm.

В папке test содержится конфиг (config.json) и данные (аудиоайлы) для теста. 

train.py и test.py были подкорректированы под задачу. 

Датасет скачивается с kaggle. Папка с финальной моделью ./final_model/ скачивается с google drive. Скрипты для скачивания приведены ниже.

Dockerfile не валиден, необходимые пакеты устанавливаются с помощью requirements.txt.

exps_part1.ipynb, exps_part1.ipynb - ноутбуки, в которых проводились эксперименты. hw5_test.ipynb - ноутбук с тестированием модели на аудиозаписях из папки test. В папке final_model хранится результат эксперимента (все хинты без первого(abs) и последнего(unfixed sinc filters)), в ходе которого был выбит нужный скор, там лежат конфиг, лог обучения и веса модели model_best.pth.

____

Устанавливать библиотеки нужно с помощью requirements.txt. Dockerfile невалидный.

Guide по установке:
```
git clone https://github.com/AnyaAkhmatova/hw_cm.git
```
Из директории ./hw_cm/ (устанавливаем библиотеки и нашу маленькую библиотечку, скачиваем датасет (приведен вариант скачивания в google colab, нужен kaggle.json, в kaggle ничего скачивать не нужно, просто добавить датасет), финальную модель (лог, конфиг, чекпоинт)):

```
pip install -r requirements.txt
pip install .

pip install kaggle
mkdir ~/.kaggle
mv ./kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d awsaf49/asvpoof-2019-dataset
rm -r ~/.kaggle
mkdir ./data
unzip -q ./asvpoof-2019-dataset.zip -d ./data/
rm -rf asvpoof-2019-dataset.zip

gdown https://drive.google.com/uc?id=1Z2XQM0sQqMq4AW9g8Z6aARDBipV6RlH1
unzip final_model.zip
rm -rf final_model.zip
```

Wandb:

```
import wandb

wandb.login()
```

Запуск train:

```
!python3 train.py -c ./hw_cm/configs/exp1.json
```

Запуск test:

```
!python3 test.py -c ./test/config.json -r ./final_model/model_best.pth
```

Комментарий: обучение запускалось в kaggle, тестирование - в google colab.

____

W&B Report: https://wandb.ai/crazy_ocean_ahead/cm_project/reports/HW5-Voice-Anti-spoofing-report--Vmlldzo2Mjc4NjE2?accessToken=hkqmq3zl5xcktbbeus2p27rntj65nmg744pqfjocr5ofvdlmkcrb4x1j4v1zlgua.

____

Бонусов нет.

____


Выполнила Ахматова Аня, группа 201.


# norfair-demos

## Установка зависимостей
```bash
pip install -r requirements.txt
```

> Если возникнут проблемы с установкой `detectron2`, то можно:
> 1. Убрать `detecron2` из `requirements.txt` и установить его потом отдельно:
>   ```bash
>   pip install 'git+https://github.com/facebookresearch/detectron2.git'
>   ```
> 2. Попробовать альтернативный вариант из [документация](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#:~:text=demo%20and%20visualization-,Build%20Detectron2%20from%20Source,-gcc%20%26%20g%2B%2B%20%E2%89%A5%205.4)

## Запуск
```bash
    python norfair_demos <detector> <video.mp4>
```
* `detector`:
  * `CircleDetector` -- использует `cv2.HoughCircles()` 
  * `BackgroundCircleDetector` -- использует подход: **background subtraction**
  * `DetectronCarDetector` -- на основе `detectron2`

> Для работы с `detecron2` изначально надо загрузить веса модели:
> ```bash
> wget https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl .nor_fair_demos/cfg_files/model_final_f10217.pkl
> ```

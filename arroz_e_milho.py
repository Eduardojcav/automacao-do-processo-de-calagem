
!pip install ultralytics --quiet

from google.colab import drive
drive.mount('/content/drive')

zip_path = '/content/drive/MyDrive/demarcacao de cacamba.v2i.yolov8.zip'

!ls /content/dataset

!unzip -q "$zip_path" -d /content/dataset

!ls /content/dataset/train/images
!ls /content/dataset/valid/images

data_yaml = """
path: /content/dataset
train: train/images
val: valid/images
names: ['cacamba']
"""

with open("/content/data.yaml", "w") as f:
    f.write(data_yaml)


from ultralytics import YOLO
model = YOLO('yolov8n.yaml')  
model.train(data="/content/data.yaml", epochs=40)

!ls "/content/drive/MyDrive/videos_teste"

!pip install ultralytics --quiet

from google.colab import drive
from ultralytics import YOLO
from IPython.display import Video, display
import os

drive.mount('/content/drive')

model = YOLO('/content/runs/detect/train7/weights/best.pt')

video_names = [
    'caminhoes.mp4',    
    'caminhoes1.mp4',
    'Caminhoes2.mp4',
    'Caminhoes3.mp4',
    'Caminhoes4.mp4',
    'Caminhoes5.mp4',
    'Caminhoes6.mp4',
    'Caminhoes7.mp4',
    'Caminhoes8.mp4',
    'Caminhoes9.mp4'
]

drive_folder = '/content/drive/MyDrive/videos_teste'

for idx, video_name in enumerate(video_names, start=1):
    video_path = os.path.join(drive_folder, video_name)
    predict_name = f'predict_video{idx}'

    print(f"\n🔍 Processando vídeo {idx}: {video_name} ...")

    results = model.predict(
        source=video_path,
        save=True,
        conf=0.5,
        project='runs/detect',
        name=predict_name
    )

    predicted_video_path = f'/content/runs/detect/{predict_name}/{video_name}'

    print(f"🎬 Resultado do vídeo {idx}:")
    display(Video(predicted_video_path, embed=True))

"""# Nova seção"""

import os
import shutil
import subprocess

video_names = [
    'caminhoes.mp4',
    'caminhoes1.mp4',
    'Caminhoes2.mp4',
    'Caminhoes3.mp4',
    'Caminhoes4.mp4',
    'Caminhoes5.mp4',
    'Caminhoes6.mp4',
    'Caminhoes7.mp4',
    'Caminhoes8.mp4',
    'Caminhoes9.mp4'
]

dest_folder = '/content/drive/MyDrive/videos_resultado'
os.makedirs(dest_folder, exist_ok=True)

for idx, original_name in enumerate(video_names, start=1):
    avi_path = f'/content/runs/detect/predict_video{idx}/{original_name.replace(".mp4", ".avi")}'
    mp4_path = f'/content/runs/detect/predict_video{idx}/{original_name}'
    target_path = os.path.join(dest_folder, original_name)

    if os.path.exists(avi_path):
        print(f'🎬 Convertendo {avi_path} para .mp4...')
        subprocess.run(['ffmpeg', '-y', '-i', avi_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '22', mp4_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(mp4_path):
            shutil.copy(mp4_path, target_path)
            print(f'✅ Copiado: {original_name}')
        else:
            print(f'⚠️ Falha ao converter para .mp4: {mp4_path}')
    else:
        print(f'⚠️ Arquivo .avi não encontrado: {avi_path}')

import warnings
from pathlib import Path

import cv2
import os

import numpy as np
from natsort import natsorted
from tkinter import Tk, filedialog
from tqdm import tqdm
import torch
from transformers import AutoModelForImageSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

def bilder_zu_video(image_folder, video_name="output_video.mp4", fps=30):
    # Bilder sammeln und sortieren
    images = [img for img in os.listdir(image_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]
    images = natsorted(images)

    if not images:
        print("Keine Bilder im gewaehlten Ordner gefunden.")
        return

    # Erste Bilddatei zur Groessenermittlung
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Fehler beim Lesen von {first_image_path}")
        return

    height, width, _ = frame.shape

    # Video-Writer vorbereiten
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    print("Erstelle Video...")

    for img_name in tqdm(images, desc="Verarbeite Bilder", unit="Bild"):
        img_path = os.path.join(image_folder, img_name)
        frame = cv2.imread(img_path)
        if frame is not None:
            video.write(frame)
        else:
            print(f"Warnung: Fehler beim Lesen von {img_path}")

    video.release()
    print(f"Video gespeichert als {video_name}")

def delete_img(ordnerpfad: Path):
    # Alle Bilddateien im Ordner sammeln und sortieren
    bild_dateien = sorted([
        f for f in ordnerpfad.iterdir()
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
    ])

    # Jedes zweite Bild löschen (beginnend bei Index 1)
    for index, bild in enumerate(bild_dateien):
        if index % 2 == 1:
            try:
                bild.unlink()
                print(f"Gelöscht: {bild}")
            except Exception as e:
                print(f"Fehler beim Löschen von {bild}: {e}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def img_segmentation(img_path: Path = Path("C:/Users/martn/Downloads/images/IMG_2017.JPG"),
                     output_path: Path = Path("C:/Users/martn/Downloads/image_with_alpha.png")):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Fehler beim Lesen von {img_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        birefnet = AutoModelForImageSegmentation.from_pretrained(
            'ZhengPeng7/BiRefNet', trust_remote_code=True
        )
        birefnet = birefnet.to(device)
        if device.type == "cuda":
            birefnet = birefnet.half()
        birefnet.eval()

    # Ergebnisbild mit Alpha-Kanal extrahieren
    result = extract_object(birefnet, imagepath=img_path)
    image_with_alpha = result[0]

    # Sicherstellen, dass es ein PIL-Bild ist, oder konvertieren
    if isinstance(image_with_alpha, np.ndarray):
        image_with_alpha = Image.fromarray(image_with_alpha)

    # Speichern als PNG mit Transparenz
    image_with_alpha.save(output_path)
    print(f"Bild mit Alpha-Kanal gespeichert unter: {output_path}")

    # Anzeigen (optional)
    plt.axis("off")
    plt.imshow(image_with_alpha)
    plt.show()

def extract_object(birefnet, imagepath):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(imagepath)
    input_images = transform_image(image).unsqueeze(0).to('cuda').half()

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    return image, mask

def main():
    img_segmentation()
    return
    # GUI-Fenster ausblenden
    root = Tk()
    root.withdraw()

    print("Bitte waehle den Ordner mit den Bildern aus.")
    folder_selected = filedialog.askdirectory(title="Bilder-Ordner auswaehlen")

    if not folder_selected:
        print("Kein Ordner ausgewaehlt. Vorgang abgebrochen.")
        return

    #delete_img(Path(folder_selected))

    video_name = os.path.join(folder_selected, "video_output.mp4")
    bilder_zu_video(folder_selected, video_name)

if __name__ == "__main__":
    main()

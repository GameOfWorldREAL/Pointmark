import h5py
import laspy
import numpy as np
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import open3d as o3d
import pye57

from src.pointmark.setup.setup import setup_UI


def save(ref_point_cloud: np.ndarray, path: Path):
    data = {"ref_point_cloud": ref_point_cloud }

    with h5py.File((path / "ref_point_cloud.h5"), "w") as file:
        for name, data_array in data.items():
            file.create_dataset(name, data=data_array, compression="gzip")

def restore(path):
    with h5py.File(path, 'r') as file:
        ref_point_cloud  = file["ref_point_cloud"][:]
    return ref_point_cloud


def las_or_e57():
    # Datei-Dialog anzeigen
    file_path = filedialog.askopenfilename(
        title="Wähle eine Punktwolken-Datei (.las oder .e57)",
        filetypes=[("Punktwolken-Dateien", "*.las *.e57"), ("LAS", "*.las"), ("E57", "*.e57")]
    )

    if not file_path:
        print("Keine Datei ausgewählt.")
        return

    extension = Path(file_path).suffix.lower()

    if extension == ".las":
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).T

    elif extension == ".e57":
        e57 = pye57.E57(file_path)

        try:
            scans_meta = e57.image_file.root()["data3D"]
            num_scans = len(scans_meta)
            print(f"{num_scans} Scan(s) in .e57-Datei gefunden.")

            all_points = []

            for idx in range(num_scans):
                print(f"Lese Scan {idx + 1} / {num_scans}")
                scan = e57.read_scan(idx)

                if not all(k in scan for k in ("cartesianX", "cartesianY", "cartesianZ")):
                    print(f"Scan {idx} übersprungen: keine gültigen Koordinaten.")
                    continue

                x = np.asarray(scan["cartesianX"])
                y = np.asarray(scan["cartesianY"])
                z = np.asarray(scan["cartesianZ"])

                if "cartesianInvalidState" in scan:
                    valid = ~np.asarray(scan["cartesianInvalidState"])
                    x, y, z = x[valid], y[valid], z[valid]

                points = np.column_stack((x, y, z))
                all_points.append(points)

            if not all_points:
                print("Keine gültigen Scandaten gefunden.")
                return

            points = np.concatenate(all_points, axis=0)

        except Exception as e:
            print(f"Fehler beim Lesen der E57-Datei: {e}")
            return

    else:
        print(f"Dateiformat {extension} wird nicht unterstützt.")
        return

    print(f"{points.shape[0]} Punkte geladen aus:\n{file_path}")
    save(points, Path(file_path).parent)


def ply():
    # GUI für Dateiauswahl (tkinter)
    root = tk.Tk()
    root.withdraw()  # Hauptfenster ausblenden

    # Dateiauswahl-Dialog anzeigen
    file_path = filedialog.askopenfilename(
        title="Wähle eine .ply-Datei",
        filetypes=[("PLY files", "*.ply")]
    )

    # Prüfen ob Datei ausgewählt wurde
    if file_path:
        # PLY-Datei lesen
        pcd = o3d.io.read_point_cloud(file_path)

        # Punkte als NumPy-Array (N, 3)
        points = np.asarray(pcd.points)

        print(f"{points.shape[0]} Punkte geladen aus:\n{file_path}")
        save(points, Path(file_path).parent)
    else:
        print("Keine Datei ausgewählt.")


def convert_e57_to_ply():
    # GUI-Dateiauswahl
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Wähle eine .e57-Datei",
        filetypes=[("E57-Dateien", "*.e57")]
    )

    if not file_path:
        print("Keine Datei ausgewählt.")
        return

    try:
        e57 = pye57.E57(file_path)
        scans_meta = e57.image_file.root()["data3D"]
        num_scans = len(scans_meta)
        print(f"{num_scans} Scan(s) in Datei gefunden.")

        points_list = []

        for idx in range(num_scans):
            print(f"Lese Scan {idx + 1} / {num_scans}")
            scan = e57.read_scan(idx)

            # Pflichtfelder prüfen
            if not all(k in scan for k in ("cartesianX", "cartesianY", "cartesianZ")):
                print(f"Scan {idx} enthält keine gültigen 3D-Koordinaten.")
                continue

            # Als NumPy-Arrays
            x, y, z = map(np.asarray, (scan["cartesianX"], scan["cartesianY"], scan["cartesianZ"]))

            # Gültige Punkte filtern (falls vorhanden)
            if "cartesianInvalidState" in scan:
                valid = ~np.asarray(scan["cartesianInvalidState"])
                x, y, z = x[valid], y[valid], z[valid]

            # Direkt stapeln
            pts = np.column_stack((x, y, z))
            points_list.append(pts)

        if not points_list:
            print("Keine gültigen Punkte gefunden.")
            return

        # Alle Punkte zusammenführen
        all_points = np.concatenate(points_list, axis=0)
        print(f"{all_points.shape[0]} Punkte geladen.")

        # PointCloud erzeugen
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))

        # Als .ply speichern
        output_path = Path(file_path).with_suffix(".ply")
        o3d.io.write_point_cloud(str(output_path), pcd)
        print(f"PLY-Datei erfolgreich gespeichert:\n{output_path}")

    except Exception as e:
        print(f"Fehler: {e}")

def main():
    setup_UI()
    las_or_e57()

if __name__ == "__main__":
    main()
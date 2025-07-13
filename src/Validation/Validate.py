import argparse
import numpy as np
from pathlib import Path
from tkinter import filedialog
from src.Validation.validationData import ValidationData
from src.general.python.filePaths import ProjectPaths, PointmarkPaths
from src.metrics.MetricData import METRICS
from src.pointmark.setup.setup import setup_UI

pointmark_paths = PointmarkPaths(Path(__file__).resolve().parents[2])

def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])  # (N, 4)
    transformed_h = (T @ points_h.T).T  # (N, 4)
    return transformed_h[:, :3]

def validate(project_path: Path, ref_path: Path):
    project_paths = ProjectPaths(project_path)
    val_data = ValidationData(pointmark_paths, project_paths)
    if val_data.has_save_data():
        val_data.restore()
    else:
        #val_data.build_ground_truth(ref_path)
        val_data.compute_distances(ref_path)
        val_data.draw_distances()
        val_data.draw_distances(dist=True)
        val_data.save()

    val_data.plot(graph=False, only_metric=METRICS.COMBINED, smoothing=0.004, bins=700, percent=99.9)
    val_data.plot(graph=False, only_metric=METRICS.COMBINED, ooi=True, smoothing=0.01, percent=99.999)

    if False:
        view_metric = METRICS.CAMERA_DISTANCE
        val_data.plot(graph=False, only_metric=view_metric, smoothing=0.004, bins = 700, percent=99.9)
        val_data.plot(normed=True, graph=False, smoothing=0.02, only_metric=view_metric, bins = 700, percent=100)
        val_data.plot(graph=False, only_metric=view_metric, ooi=True, smoothing=0.01, percent=99.999)
        val_data.plot(normed=True, graph=False, only_metric=view_metric, ooi=True, smoothing=0.01, percent=100)

    if False:
        view_metric = METRICS.OBSERVATION_COUNT
        val_data.plot(graph=False, only_metric=view_metric, percent=99.95)
        val_data.plot(normed=True, graph=False, smoothing=0.03, only_metric=view_metric, percent=100)
        val_data.plot(graph=False, only_metric=view_metric, ooi=True, smoothing=0.03, percent=100)
        val_data.plot(normed=True, graph=False, only_metric=view_metric, ooi=True, smoothing=0.03, percent=100)

    if False:
        view_metric = METRICS.TRIANGULATION_UNCERTAINTY
        val_data.plot(graph=False, only_metric=view_metric)
        val_data.plot(normed=True, graph=False, smoothing=0.03, only_metric=view_metric)
        val_data.plot(graph=False, only_metric=view_metric, ooi=True, smoothing=0.03)
        val_data.plot(normed=True, graph=False, only_metric=view_metric, ooi=True, smoothing=0.03)

    if False:
        view_metric = METRICS.ANGLE_OF_INCIDENCE
        val_data.plot(graph=False, only_metric=view_metric)
        val_data.plot(normed=True, graph=False, smoothing=0.03, only_metric=view_metric)
        val_data.plot(graph=False, only_metric=view_metric, ooi=True, smoothing=0.03, percent=99.999)
        val_data.plot(normed=True, graph=False, only_metric=view_metric, ooi=True, smoothing=0.03, percent=99.999)

    if False:
        view_metric = METRICS.PROJECTION_ERROR
        val_data.plot(graph=False, only_metric=view_metric, bins = 1000, percent=99)
        val_data.plot(normed=True, graph=False, smoothing=0.03, only_metric=view_metric)
        val_data.plot(graph=False, only_metric=view_metric, ooi=True, smoothing=0.03)
        val_data.plot(normed=True, graph=False, only_metric=view_metric, ooi=True, smoothing=0.03)

    if False:
        view_metric = METRICS.DENSITY
        val_data.plot(graph=False, only_metric=view_metric,smoothing=0.002, bins = 5000)
        val_data.plot(normed=True, graph=False, smoothing=0.01, only_metric=view_metric)
        val_data.plot(graph=False, only_metric=view_metric, ooi=True, smoothing=0.005, bins = 1000, no_green=True, no_red=True)
        val_data.plot(normed=True, graph=False, only_metric=view_metric, ooi=True, smoothing=0.01, no_green=True, no_red=True)

    if False:
        view_metric = METRICS.DISTANCE_TO_EDGE
        val_data.plot(graph=False, only_metric=view_metric, percent=99.9)
        val_data.plot(normed=True, graph=False, smoothing=0.03, only_metric=view_metric)
        val_data.plot(graph=False, only_metric=view_metric, ooi=True, smoothing=0.03)
        val_data.plot(normed=True, graph=False, only_metric=view_metric, ooi=True, smoothing=0.03)

    if False:
        view_metric = METRICS.BRIGHTNESS
        val_data.plot(graph=False, only_metric=view_metric, no_green=True, percent=99.9)
        val_data.plot(normed=True, graph=False, smoothing=0.03, only_metric=view_metric)
        val_data.plot(graph=False, only_metric=view_metric, ooi=True, smoothing=0.03, no_green=True)
        val_data.plot(normed=True, graph=False, only_metric=view_metric, ooi=True, smoothing=0.03)

    if False:
        view_metric = METRICS.DARKNESS
        val_data.plot(graph=False, only_metric=view_metric, no_green=True)
        val_data.plot(normed=True, graph=False, smoothing=0.03, only_metric=view_metric)
        val_data.plot(graph=False, only_metric=view_metric, ooi=True, smoothing=0.03)
        val_data.plot(normed=True, graph=False, only_metric=view_metric, ooi=True, smoothing=0.03)

    if False:
        view_metric = METRICS.PRECISION
        val_data.plot(graph=False, only_metric=view_metric, percent=99.9, no_green=True, no_red=True)
        val_data.plot(normed=True, graph=False, smoothing=0.03, only_metric=view_metric, percent=100, no_green=True, no_red=False, bins = 1000)
        val_data.plot(graph=False, only_metric=view_metric, ooi=True, smoothing=0.03, no_green=True, no_red=True)
        val_data.plot(normed=True, graph=False, only_metric=view_metric, ooi=True, smoothing=0.03, no_green=True, no_red=True)

    if False:
        view_metric = METRICS.IMG_QUALITY
        val_data.plot(graph=False, only_metric=view_metric)
        val_data.plot(normed=True, graph=False, smoothing=0.05, only_metric=view_metric)
        val_data.plot(graph=False, only_metric=view_metric, ooi=True, smoothing=0.03)
        val_data.plot(normed=True, graph=False, only_metric=view_metric, ooi=True, smoothing=0.05)

def main():
    parser = argparse.ArgumentParser(description="validation")
    parser.add_argument("-r", "--reference",
                        help="path to .h5-reference file",
                        type=Path)
    parser.add_argument("-p", "--project",
                        help="path to project folder",
                        type=Path)
    args = parser.parse_args()

    setup_UI()

    if args.reference is not None:
        ref_path = args.reference
    else:
        ref_path = Path(filedialog.askopenfilename(
            title="choose reference file:"
        ))
    if args.project is not None:
        project_path = args.project
    else:
        project_path = Path(filedialog.askdirectory(
            title="choose project folder:",
            initialdir=ref_path.parent
        ))

    print("reference:", ref_path)
    print("project:", project_path)
    print("-----------------------------------------")

    validate(project_path, ref_path)


if __name__ == "__main__":
    main()

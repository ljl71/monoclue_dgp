import argparse
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.datasets.kitti.depth_error_eval import evaluate_depth_errors
from lib.datasets.kitti.depth_error_eval import format_depth_error_table
from lib.datasets.kitti.depth_error_eval import read_split_file
from lib.datasets.kitti.depth_error_eval import write_depth_error_csv


def parse_result_spec(spec):
    if "=" in spec:
        method, result_dir = spec.split("=", 1)
    else:
        method, result_dir = os.path.basename(os.path.normpath(spec)), spec
    return method, result_dir


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KITTI-format monocular 3D depth errors by distance range.")
    parser.add_argument("--label-dir", required=True, help="GT label_2 directory in KITTI format.")
    parser.add_argument("--result-dir", action="append", required=True,
                        help="Prediction directory, or METHOD=DIR. Can be repeated for multiple methods.")
    parser.add_argument("--dataset-name", default="KITTI", help="Dataset name shown in the output table.")
    parser.add_argument("--split-file", default=None, help="Optional ImageSets/<split>.txt file.")
    parser.add_argument("--classes", default="Car", help="Comma-separated classes, e.g. Car or Car,Pedestrian,Cyclist.")
    parser.add_argument("--bins", default="0,20,40,80", help="Comma-separated depth bin edges in meters.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="2D IoU threshold for GT-pred matching.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Prediction score threshold.")
    parser.add_argument("--output-csv", default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    image_ids = read_split_file(args.split_file) if args.split_file else None
    classes = [item.strip() for item in args.classes.split(",") if item.strip()]

    all_rows = []
    for spec in args.result_dir:
        method, result_dir = parse_result_spec(spec)
        rows = evaluate_depth_errors(label_dir=args.label_dir,
                                     result_dir=result_dir,
                                     image_ids=image_ids,
                                     classes=classes,
                                     bins=args.bins,
                                     iou_threshold=args.iou_threshold,
                                     score_threshold=args.score_threshold)
        for row in rows:
            row["method"] = method
            row["dataset"] = args.dataset_name
        all_rows.extend(rows)

    print(format_depth_error_table(all_rows, include_method=True))
    if args.output_csv:
        write_depth_error_csv(all_rows, args.output_csv)
        print("\nSaved CSV to {}".format(args.output_csv))


if __name__ == "__main__":
    main()

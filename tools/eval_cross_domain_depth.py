import argparse
import copy
import datetime
import os
import sys
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.datasets.kitti.depth_error_eval import evaluate_depth_errors
from lib.datasets.kitti.depth_error_eval import format_depth_error_table
from lib.datasets.kitti.depth_error_eval import format_image_id
from lib.datasets.kitti.depth_error_eval import write_depth_error_csv


def parse_dataset_spec(spec):
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) not in [3, 4]:
        raise ValueError("--dataset must be NAME,ROOT_DIR,SPLIT[,IMAGE_EXT]")
    name, root_dir, split = parts[:3]
    image_ext = parts[3] if len(parts) == 4 else None
    return name, root_dir, split, image_ext


def save_results(results, class_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for img_id, detections in results.items():
        output_path = os.path.join(output_dir, "{}.txt".format(format_image_id(img_id)))
        with open(output_path, "w") as f:
            for det in detections:
                cls_name = class_name[int(det[0])]
                f.write("{} 0.0 0".format(cls_name))
                for value in det[1:]:
                    f.write(" {:.2f}".format(value))
                f.write("\n")


def run_inference(model, dataloader, device, threshold, topk, output_dir):
    torch.set_grad_enabled(False)
    model.eval()

    results = {}
    infer_time = 0.0
    progress_bar = tqdm.tqdm(total=len(dataloader), dynamic_ncols=True, leave=True)
    for inputs, calibs, targets, info in dataloader:
        inputs = inputs.to(device)
        calibs = calibs.to(device)
        img_sizes = info["img_size"].to(device)

        start_time = time.time()
        outputs = model(inputs, calibs, img_sizes, dn_args=0)
        infer_time += time.time() - start_time

        dets = extract_dets_from_outputs(outputs=outputs,
                                         K=dataloader.dataset.max_objs,
                                         topk=topk)
        dets = dets.detach().cpu().numpy()
        batch_calibs = [dataloader.dataset.get_calib(index) for index in info["img_id"]]
        info_np = {key: value.detach().cpu().numpy() for key, value in info.items()}
        dets = decode_detections(dets=dets,
                                 info=info_np,
                                 calibs=batch_calibs,
                                 cls_mean_size=dataloader.dataset.cls_mean_size,
                                 threshold=threshold)
        results.update(dets)
        progress_bar.update()
    progress_bar.close()

    save_results(results, dataloader.dataset.class_name, output_dir)
    return infer_time / max(len(dataloader), 1)


def build_dataset_cfg(base_cfg, root_dir, split, image_ext, batch_size):
    dataset_cfg = copy.deepcopy(base_cfg)
    dataset_cfg["type"] = dataset_cfg.get("type", "KITTI")
    dataset_cfg["root_dir"] = root_dir
    dataset_cfg["test_split"] = split
    dataset_cfg["load_sam"] = False
    dataset_cfg["random_flip"] = 0.0
    dataset_cfg["random_crop"] = 0.0
    dataset_cfg["random_mixup3d"] = 0.0
    if batch_size is not None:
        dataset_cfg["batch_size"] = batch_size
    if image_ext:
        dataset_cfg["image_ext"] = image_ext
    return dataset_cfg


def main():
    parser = argparse.ArgumentParser(
        description="Run a KITTI-trained MonoCLUE checkpoint on KITTI-like target datasets and report depth MAE.")
    parser.add_argument("--config", required=True, help="Model config yaml.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path trained on KITTI train.")
    parser.add_argument("--dataset", action="append", required=True,
                        help="Dataset spec NAME,ROOT_DIR,SPLIT[,IMAGE_EXT]. Repeat for KITTI and nuScenes frontal.")
    parser.add_argument("--output-root", default=None, help="Root directory for predictions and CSV.")
    parser.add_argument("--run-name", default=None, help="Subdirectory name. Defaults to a timestamp.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for evaluation.")
    parser.add_argument("--workers", type=int, default=7, help="Dataloader workers.")
    parser.add_argument("--threshold", type=float, default=None, help="Detection confidence threshold.")
    parser.add_argument("--topk", type=int, default=None, help="Top-k detections per image before score filtering.")
    parser.add_argument("--classes", default="Car", help="Comma-separated classes.")
    parser.add_argument("--bins", default="0,20,40,80", help="Comma-separated depth bin edges in meters.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="2D IoU threshold for GT-pred matching.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Prediction score threshold for depth MAE.")
    parser.add_argument("--non-strict", action="store_true", help="Load checkpoint with strict=False.")
    args = parser.parse_args()

    global torch, tqdm, decode_detections, extract_dets_from_outputs
    import torch
    import tqdm
    import yaml
    from lib.helpers.dataloader_helper import build_test_dataloader
    from lib.helpers.decode_helper import decode_detections
    from lib.helpers.decode_helper import extract_dets_from_outputs
    from lib.helpers.model_helper import build_model
    from lib.helpers.save_helper import load_checkpoint
    from lib.helpers.utils_helper import create_logger
    from lib.helpers.utils_helper import set_random_seed

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    set_random_seed(cfg.get("random_seed", 444))

    model_name = cfg["model_name"]
    run_name = args.run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or os.path.join("./", cfg["trainer"]["save_path"], model_name, "cross_domain_depth")
    output_root = os.path.join(output_root, run_name)
    os.makedirs(output_root, exist_ok=True)

    log_file = os.path.join(output_root, "eval.log")
    logger = create_logger(log_file)
    logger.info("==> Output root: {}".format(output_root))

    model, _ = build_model(cfg["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    load_checkpoint(model=model,
                    optimizer=None,
                    filename=args.checkpoint,
                    map_location=device,
                    logger=logger,
                    strict=not args.non_strict)

    threshold = args.threshold if args.threshold is not None else cfg["tester"].get("threshold", 0.2)
    topk = args.topk if args.topk is not None else cfg["tester"].get("topk", 50)
    classes = [item.strip() for item in args.classes.split(",") if item.strip()]

    all_rows = []
    for spec in args.dataset:
        dataset_name, root_dir, split, image_ext = parse_dataset_spec(spec)
        dataset_cfg = build_dataset_cfg(cfg["dataset"], root_dir, split, image_ext, args.batch_size)
        logger.info("==> Building {} from {} split {}".format(dataset_name, root_dir, split))
        dataloader = build_test_dataloader(dataset_cfg, workers=args.workers, SAM=False)

        result_dir = os.path.join(output_root, dataset_name, "data")
        avg_infer_time = run_inference(model=model,
                                       dataloader=dataloader,
                                       device=device,
                                       threshold=threshold,
                                       topk=topk,
                                       output_dir=result_dir)
        logger.info("==> {} inference: {:.4f}s / batch".format(dataset_name, avg_infer_time))

        rows = evaluate_depth_errors(label_dir=dataloader.dataset.label_dir,
                                     result_dir=result_dir,
                                     image_ids=dataloader.dataset.idx_list,
                                     classes=classes,
                                     bins=args.bins,
                                     iou_threshold=args.iou_threshold,
                                     score_threshold=args.score_threshold)
        for row in rows:
            row["method"] = model_name
            row["dataset"] = dataset_name
        all_rows.extend(rows)

    table = format_depth_error_table(all_rows, include_method=True)
    logger.info("\n" + table)
    print(table)

    csv_path = os.path.join(output_root, "depth_errors.csv")
    write_depth_error_csv(all_rows, csv_path)
    logger.info("==> Saved CSV to {}".format(csv_path))
    print("\nSaved CSV to {}".format(csv_path))


if __name__ == "__main__":
    main()

import os
import tqdm
import shutil

import torch
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from lib.datasets.kitti.depth_error_eval import evaluate_depth_errors
from lib.datasets.kitti.depth_error_eval import format_image_id
from lib.datasets.kitti.depth_error_eval import format_depth_error_table
from lib.datasets.kitti.depth_error_eval import write_depth_error_csv
import time

BOLD  = "\033[1m"
BLUE  = "\033[34m"
CYAN  = "\033[36m"
RED   = "\033[31m"
RESET = "\033[0m"

class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, train_cfg=None, model_name='monoclue'):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.train_cfg = train_cfg
        self.model_name = model_name

    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        # test a single checkpoint
        if self.cfg['mode'] == 'single' or not self.train_cfg["save_all"]:
            if self.train_cfg["save_all"]:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_epoch_{}.pth".format(self.cfg['checkpoint']))
            else:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")

            assert os.path.exists(checkpoint_path)
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=checkpoint_path,
                            map_location=self.device,
                            logger=self.logger)
            self.model.to(self.device)
            self.inference()
            self.evaluate()

        # test all checkpoints in the given dir
        elif self.cfg['mode'] == 'all' and self.train_cfg["save_all"]:
            start_epoch = int(self.cfg['checkpoint'])
            checkpoints_list = []
            for _, _, files in os.walk(self.output_dir):
                for f in files:
                    if f.endswith(".pth") and int(f[17:-4]) >= start_epoch:
                        checkpoints_list.append(os.path.join(self.output_dir, f))
            checkpoints_list.sort(key=os.path.getmtime)

            for checkpoint in checkpoints_list:
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=checkpoint,
                                map_location=self.device,
                                logger=self.logger)
                self.model.to(self.device)
                self.inference()
                self.evaluate()

    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        progress_bar = tqdm.tqdm(total=len(self.dataloader), dynamic_ncols=True, leave=True)
        model_infer_time = 0
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)

            start_time = time.time()
            ###dn
            outputs = self.model(inputs, calibs, img_sizes, dn_args = 0)
            ###
            end_time = time.time()
            model_infer_time += end_time - start_time

            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs, topk=self.cfg['topk'])

            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg.get('threshold', 0.2))

            results.update(dets)

            progress_bar.set_description(
                f"{BOLD}{BLUE}Evaluation Progress{RESET} | "f"{BOLD}{CYAN}Iter{RESET}"
            )

            progress_bar.update()

        print("inference on {} images by {}/per image".format(
            len(self.dataloader), model_infer_time / len(self.dataloader)))

        progress_bar.close()

        # save the result for evaluation.
        self.logger.info('==> Saving ...')
        self.save_results(results)

    def save_results(self, results):
        output_dir = os.path.join(self.output_dir, 'outputs', 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if self.dataset_type in ['KITTI', 'KITTI_LIKE', 'NUSCENES_FRONT'] or not hasattr(self.dataloader.dataset, 'get_sensor_modality'):
                output_path = os.path.join(output_dir, '{}.txt'.format(format_image_id(img_id)))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id),
                                           self.dataloader.dataset.get_sample_token(img_id) + '.txt')

            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()

    def evaluate(self):
        results_dir = os.path.join(self.output_dir, 'outputs', 'data')
        assert os.path.exists(results_dir)
        result = self.dataloader.dataset.eval(results_dir=results_dir, logger=self.logger)
        self.evaluate_depth_error(results_dir)
        return result

    def evaluate_depth_error(self, results_dir):
        depth_cfg = self.cfg.get('depth_error', None)
        if not depth_cfg or not depth_cfg.get('enabled', False):
            return
        if not hasattr(self.dataloader.dataset, 'label_dir'):
            self.logger.info('==> Depth error skipped: dataset has no label_dir')
            return

        rows = evaluate_depth_errors(
            label_dir=self.dataloader.dataset.label_dir,
            result_dir=results_dir,
            image_ids=self.dataloader.dataset.idx_list,
            classes=depth_cfg.get('classes', ['Car']),
            bins=depth_cfg.get('bins', [0, 20, 40, 80]),
            iou_threshold=depth_cfg.get('iou_threshold', 0.5),
            score_threshold=depth_cfg.get('score_threshold', 0.0))
        for row in rows:
            row['method'] = self.model_name
            row['dataset'] = getattr(self.dataloader.dataset, 'split', 'eval')
        self.logger.info('==> Evaluating depth error (matched 2D IoU) ...')
        self.logger.info('\n' + format_depth_error_table(rows, include_method=True))

        csv_path = depth_cfg.get('output_csv', None)
        if csv_path:
            write_depth_error_csv(rows, csv_path)
            self.logger.info('==> Depth error CSV saved to {}'.format(csv_path))

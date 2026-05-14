import argparse
import copy
import os

import yaml


def deep_update(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def sam_enabled(cfg):
    model_cfg = cfg.get('model', {})
    return (
        model_cfg.get('region_supervision', 'sam') == 'sam'
        or model_cfg.get('depth_map_supervision', 'sam') == 'sam'
    )


def make_cfg(base_cfg, name, patch):
    cfg = copy.deepcopy(base_cfg)
    cfg['model_name'] = name
    deep_update(cfg, patch)
    cfg.setdefault('dataset', {})['load_sam'] = sam_enabled(cfg)
    return cfg


def write_cfg(path, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def build_experiments(base_cfg):
    common_query = {
        'use_query_initializer': True,
        'query_initializer': {
            'use_foreground': True,
            'use_background': True,
            'use_scene_memory': True,
            'use_relocalization': True,
            'fg_num_clusters': 10,
            'bg_num_clusters': 3,
        },
    }

    experiments = {}

    # SAM supervision split: box means the original coarse 2D-box target.
    experiments['sam_box_region_box_depth'] = {
        'model': {
            'region_supervision': 'box',
            'depth_map_supervision': 'box',
        },
    }
    experiments['sam_region_only'] = {
        'model': {
            'region_supervision': 'sam',
            'depth_map_supervision': 'box',
        },
    }
    experiments['sam_depth_only'] = {
        'model': {
            'region_supervision': 'box',
            'depth_map_supervision': 'sam',
        },
    }
    experiments['sam_region_depth'] = {
        'model': {
            'region_supervision': 'sam',
            'depth_map_supervision': 'sam',
        },
    }

    # Attention comparison for Table 4.
    experiments['attention_se'] = {
        'model': {
            'region_attention': 'se',
        },
    }
    experiments['attention_eca'] = {
        'model': {
            'region_attention': 'eca',
        },
    }

    # Query prototype component ablation for Table 5.
    experiments['proto_none'] = {
        'model': {
            'use_query_initializer': False,
        },
    }
    experiments['proto_fg'] = {
        'model': {
            **common_query,
            'query_initializer': {
                **common_query['query_initializer'],
                'use_background': False,
                'use_scene_memory': False,
                'use_relocalization': False,
            },
        },
    }
    experiments['proto_fg_bg'] = {
        'model': {
            **common_query,
            'query_initializer': {
                **common_query['query_initializer'],
                'use_scene_memory': False,
                'use_relocalization': False,
            },
        },
    }
    experiments['proto_fg_bg_scene'] = {
        'model': {
            **common_query,
            'query_initializer': {
                **common_query['query_initializer'],
                'use_relocalization': False,
            },
        },
    }
    experiments['proto_full'] = {
        'model': common_query,
    }

    # Foreground prototype count sensitivity.
    for num_clusters in (5, 10, 15):
        experiments[f'proto_num_{num_clusters}'] = {
            'model': {
                **common_query,
                'query_initializer': {
                    **common_query['query_initializer'],
                    'fg_num_clusters': num_clusters,
                },
            },
        }

    return {name: make_cfg(base_cfg, name, patch) for name, patch in experiments.items()}


def main():
    parser = argparse.ArgumentParser(description='Generate MonoCLUE-DGP ablation config files.')
    parser.add_argument('--base', default='./configs/monoclue.yaml', help='Base yaml config.')
    parser.add_argument('--out-dir', default='./configs/ablations', help='Output directory.')
    parser.add_argument('--test-split', default='val', help='Evaluation split for ablation configs.')
    args = parser.parse_args()

    with open(args.base, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)
    if args.test_split:
        base_cfg.setdefault('dataset', {})['test_split'] = args.test_split

    experiments = build_experiments(base_cfg)
    for name, cfg in experiments.items():
        out_path = os.path.join(args.out_dir, f'{name}.yaml')
        write_cfg(out_path, cfg)
        print(out_path)


if __name__ == '__main__':
    main()

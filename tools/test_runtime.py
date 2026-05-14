from thop import profile
import torch
import os
import yaml
import sys
import time
import warnings
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


CUDA_VISIBLE_DEVICES = 0

parser = argparse.ArgumentParser(description='Profile params, FLOPs, and runtime for an ablation config.')
parser.add_argument('--config', default='./configs/monoclue.yaml', help='Path to a yaml config file.')
parser.add_argument('--warmup', type=int, default=10)
parser.add_argument('--iters', type=int, default=100)
args = parser.parse_args()

cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

is_monodgp = cfg.get('model_name') == 'monodgp'
if is_monodgp:
    from lib.models.monodgp import build_monodgp
    model, loss = build_monodgp(cfg['model'])
else:
    from lib.models.monoclue import build_monoclue
    model, loss = build_monoclue(cfg['model'])
model = model.to(device)
model.eval() 

input_img = torch.randn(1, 3, 384, 1280).to(device)  
calib = torch.zeros(1, 3, 4, device=device)
calib[:, 0, 0] = 721.5377
calib[:, 1, 1] = 721.5377
calib[:, 0, 2] = 609.5593
calib[:, 1, 2] = 172.8540
calib[:, 2, 2] = 1.0
sizes = torch.tensor([[1280., 384.]], device=device)
input_size = (3, 384, 1280) 

forward_inputs = (input_img, calib, None, sizes) if is_monodgp else (input_img, calib, sizes)


# warm up
with torch.no_grad():
    for _ in range(args.warmup):
        _ = model(*forward_inputs)
if torch.cuda.is_available():
    torch.cuda.synchronize()

# test runtime(ms)
start_time = time.time()
with torch.no_grad():
    for _ in range(args.iters):
        _ = model(*forward_inputs)
if torch.cuda.is_available():
    torch.cuda.synchronize()
end_time = time.time()

inference_time = (end_time - start_time) / args.iters * 1000  


# precise test
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)
# torch.cuda.synchronize()
#
# start_event.record()
# for _ in range(100):
#     _ = model(input_img, calib, None, sizes)
# end_event.record()
# torch.cuda.synchronize()
#
# inference_time = start_event.elapsed_time(end_event) / 100


# test params and flops
flops, params = profile(model, inputs=forward_inputs)

print(f"Inference Time: {inference_time:.2f} ms")
print(f"Params: {params / 1e6:.2f} M")
print(f"FLOPS: {flops / 1e9:.2f} G")

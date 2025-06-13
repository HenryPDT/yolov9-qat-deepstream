import os
import onnx
import torch
import torch.nn as nn
import warnings

import utils.tal.anchor_generator as _m

# QAT support imports
try:
    import models.quantize as quantize
    from pytorch_quantization import nn as quant_nn
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False
    print("Warning: QAT modules not available. Only standard models supported.")


def _dist2bbox(distance, anchor_points, xywh=False, dim=-1):
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat((x1y1, x2y2), dim)


_m.dist2bbox.__code__ = _dist2bbox.__code__


class DeepStreamOutputDual(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x[1].transpose(1, 2)
        boxes = x[:, :, :4]
        scores, labels = torch.max(x[:, :, 4:], dim=-1, keepdim=True)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.transpose(1, 2)
        boxes = x[:, :, :4]
        scores, labels = torch.max(x[:, :, 4:], dim=-1, keepdim=True)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


def detect_qat_model(model):
    """Detect if model is QAT (Quantization Aware Training)"""
    if not QAT_AVAILABLE:
        return False
    
    try:
        for i in range(0, len(model.model)):
            layer = model.model[i]
            if quantize.have_quantizer(layer):
                return True
    except:
        pass
    return False


def yolov9_export(weights, device, inplace=True, fuse=True):
    ckpt = torch.load(weights, map_location='cpu')
    ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()
    if not hasattr(ckpt, 'stride'):
        ckpt.stride = torch.tensor([32.])
    if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
        ckpt.names = dict(enumerate(ckpt.names))
    model = ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval()
    for m in model.modules():
        t = type(m)
        if t.__name__ in ('Hardswish', 'LeakyReLU', 'ReLU', 'ReLU6', 'SiLU', 'Detect', 'Model'):
            m.inplace = inplace
        elif t.__name__ == 'Upsample' and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None
    model.eval()
    head = 'Detect'
    for k, m in model.named_modules():
        if m.__class__.__name__ in ('Detect', 'DDetect', 'DualDetect', 'DualDDetect'):
            m.inplace = False
            m.dynamic = False
            m.export = True
            head = m.__class__.__name__
    return model, head


def suppress_warnings():
    import warnings
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=ResourceWarning)


def export_onnx_with_qat_support(model, input_tensor, output_file, opset_version, dynamic_axes, is_qat):
    """Export ONNX with QAT support if needed"""
    if is_qat and QAT_AVAILABLE:
        print('QAT Model Detected - Using quantization-aware export')
        warnings.filterwarnings("ignore")
        quant_nn.TensorQuantizer.use_fb_fake_quant = True
        model.eval()
        quantize.initialize()
        quantize.replace_custom_module_forward(model)
        
        with torch.no_grad():
            torch.onnx.export(
                model, 
                input_tensor, 
                output_file, 
                verbose=False, 
                opset_version=opset_version, 
                do_constant_folding=True,
                input_names=['input'], 
                output_names=['output'], 
                dynamic_axes=dynamic_axes
            )
        quant_nn.TensorQuantizer.use_fb_fake_quant = False
    else:
        print('Standard Model - Using regular export')
        torch.onnx.export(
            model, 
            input_tensor, 
            output_file, 
            verbose=False, 
            opset_version=opset_version, 
            do_constant_folding=True,
            input_names=['input'], 
            output_names=['output'], 
            dynamic_axes=dynamic_axes
        )


def main(args):
    suppress_warnings()

    print(f'\nStarting: {args.weights}')
    print('Opening YOLOv9 model')

    device = torch.device('cpu')
    model, head = yolov9_export(args.weights, device)

    # Check if model is QAT
    is_qat = detect_qat_model(model)
    if is_qat:
        print('✅ QAT Model Detected')
    else:
        print('📝 Standard Model (Non-QAT)')

    if len(model.names.keys()) > 0:
        print('Creating labels.txt file')
        with open('labels.txt', 'w', encoding='utf-8') as f:
            for name in model.names.values():
                f.write(f'{name}\n')

    # Add DeepStream output layer
    if head in ('Detect', 'DDetect'):
        model = nn.Sequential(model, DeepStreamOutput())
    else:
        model = nn.Sequential(model, DeepStreamOutputDual())

    img_size = args.size * 2 if len(args.size) == 1 else args.size
    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)
    onnx_output_file = f'{args.weights}.onnx'

    dynamic_axes = {
        'input': {0: 'batch'},
        'output': {0: 'batch'}
    } if args.dynamic else None

    print('Exporting the model to ONNX')
    export_onnx_with_qat_support(
        model, 
        onnx_input_im, 
        onnx_output_file, 
        args.opset, 
        dynamic_axes, 
        is_qat
    )

    if args.simplify:
        print('Simplifying the ONNX model')
        import onnxslim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print(f'Done: {onnx_output_file}')
    print(f'Model Type: {"QAT" if is_qat else "Standard"}')
    print(f'Head Type: {head}')
    print(f'DeepStream Compatible: ✅\n')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DeepStream YOLOv9 conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Static batch-size')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and static batch-size at same time')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
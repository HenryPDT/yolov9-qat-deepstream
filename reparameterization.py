import torch
from models.yolo import Model
import argparse


def main(args):
    device = torch.device(args.device)
    model = Model(args.cfg, ch=3, nc=args.classes_num, anchors=3)
    #model = model.half()
    model = model.to(device)
    _ = model.eval()
    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    model.names = ckpt['model'].names
    model.nc = ckpt['model'].nc
    idx = 0
    for k, v in model.state_dict().items():
        if "model.{}.".format(idx) in k:
            if args.model == "s" and idx < 22:
                # YOLOv9-S: direct mapping (no offset)
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif (args.model == "c" and idx < 22) or (args.model == "m" and idx < 22):
                # YOLOv9-C and YOLOv9-M: offset by +1
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx + 1))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif (args.model == "e" and idx < 29):
                # YOLOv9-E: direct mapping for layers < 29
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif args.model == 'e' and idx < 42:
                # YOLOv9-E: offset by +7 for layers 29-41
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx + 7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                if args.model == "s":
                    # YOLOv9-S: cv2 -> cv4 with +7 offset
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx + 7))
                elif args.model in ["c", "m"]:
                    # YOLOv9-C and YOLOv9-M: cv2 -> cv4 with +16 offset
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx + 16))
                else:  # args.model == "e"
                    # YOLOv9-E: cv2 -> cv4 with +7 offset
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx + 7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                if args.model == "s":
                    # YOLOv9-S: cv3 -> cv5 with +7 offset
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx + 7))
                elif args.model in ["c", "m"]:
                    # YOLOv9-C and YOLOv9-M: cv3 -> cv5 with +16 offset
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx + 16))
                else:  # args.model == "e"
                    # YOLOv9-E: cv3 -> cv5 with +7 offset
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx + 7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                if args.model == "s":
                    # YOLOv9-S: dfl -> dfl2 with +7 offset
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx + 7))
                elif args.model in ["c", "m"]:
                    # YOLOv9-C and YOLOv9-M: dfl -> dfl2 with +16 offset
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx + 16))
                else:  # args.model == "e"
                    # YOLOv9-E: dfl -> dfl2 with +7 offset
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx + 7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
        else:
            while True:
                idx += 1
                if "model.{}.".format(idx) in k:
                    break
            if args.model == "s" and idx < 22:
                # YOLOv9-S: direct mapping (no offset)
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif (args.model == "c" and idx < 22) or (args.model == "m" and idx < 22):
                # YOLOv9-C and YOLOv9-M: offset by +1
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx + 1))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif (args.model == "e" and idx < 29):
                # YOLOv9-E: direct mapping for layers < 29
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif args.model == 'e' and idx < 42:
                # YOLOv9-E: offset by +7 for layers 29-41
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx + 7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                if args.model == "s":
                    # YOLOv9-S: cv2 -> cv4 with +7 offset
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx + 7))
                elif args.model in ["c", "m"]:
                    # YOLOv9-C and YOLOv9-M: cv2 -> cv4 with +16 offset
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx + 16))
                else:  # args.model == "e"
                    # YOLOv9-E: cv2 -> cv4 with +7 offset
                    kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx + 7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                if args.model == "s":
                    # YOLOv9-S: cv3 -> cv5 with +7 offset
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx + 7))
                elif args.model in ["c", "m"]:
                    # YOLOv9-C and YOLOv9-M: cv3 -> cv5 with +16 offset
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx + 16))
                else:  # args.model == "e"
                    # YOLOv9-E: cv3 -> cv5 with +7 offset
                    kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx + 7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                if args.model == "s":
                    # YOLOv9-S: dfl -> dfl2 with +7 offset
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx + 7))
                elif args.model in ["c", "m"]:
                    # YOLOv9-C and YOLOv9-M: dfl -> dfl2 with +16 offset
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx + 16))
                else:  # args.model == "e"
                    # YOLOv9-E: dfl -> dfl2 with +7 offset
                    kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx + 7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                print(k, "perfectly matched!!")
    _ = model.eval()

    m_ckpt = {'model': model.half(),
              'optimizer': None,
              'best_fitness': None,
              'ema': None,
              'updates': None,
              'opt': None,
              'git': None,
              'date': None,
              'epoch': -1}
    torch.save(m_ckpt, args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='../models/detect/gelan-c.yaml', help='model.yaml path')
    parser.add_argument('--model', type=str, default='c', help='convert model type (s, m, c, or e)')
    parser.add_argument('--weights', type=str, default='./yolov9-c.pt', help='weights path')
    parser.add_argument('--device', default='cpu', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--classes_num', default=80, type=int, help='number of classes')
    parser.add_argument('--save', default='./yolov9-c-converted.pt', type=str, help='save path')
    args = parser.parse_args()
    main(args)

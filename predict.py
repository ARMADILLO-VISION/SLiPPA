if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to model state dictionary")
    parser.add_argument("--arch", help="model architecture")
    parser.add_argument("--dataset", help="testing dataset (default: P2ILF)")
    args = parser.parse_args()

    possible_arch = ["unet", "unetpp", "unet3p", "resunetpp", "deeplabv3p"]
    possible_dataset = ["P2ILF", "L3D"]

    if args.arch not in possible_arch:
        print("Unknown model architecture!")
        sys.exit(1)
    
    if args.dataset not in possible_dataset:
        print("Unknown dataset!")
        sys.exit(1)
    
    print("Initialising...")

    import torch
    import cv2 as cv
    import random

    from torchvision.transforms import Compose
    from segmentation_models_pytorch import Unet, UnetPlusPlus, DeepLabV3Plus

    from utils.dataset import P2ILF, L3D
    from utils.transform import Resize, ToTensor
    from utils.visualise import image_mask_overlay
    from models.unet3p.model import Unet3Plus
    from models.resunetpp.model import ResUnetPlusPlus
    from utils.postprocess import process_mask


    keys = {
        "P2ILF": P2ILF,
        "L3D": L3D
    }

    test_data = keys[args.dataset](
        "test",
        Compose([
            Resize(320, 416),
            ToTensor()
        ])
    )

    MODELS = {
        "unet": Unet(classes=4, encoder_weights="imagenet" if not args.no_pretrain else None),
        "unetpp": UnetPlusPlus(classes=4, encoder_weights="imagenet" if not args.no_pretrain else None),
        "unet3p": Unet3Plus(classes=4, encoder_weights="imagenet" if not args.no_pretrain else None),
        "resunetpp": ResUnetPlusPlus(classes=4, encoder_weights="imagenet" if not args.no_pretrain else None),
        "deeplabv3p": DeepLabV3Plus(classes=4, encoder_weights="imagenet" if not args.no_pretrain else None)
    }

    model = MODELS[args.arch]
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model.to(device)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model.eval()

    while True:
        try:
            image = random.randint(0, len(test_data) - 1)
            print(f"Testing on image {image}")
            x, y = test_data[image]
            with torch.no_grad():
                prediction = model(x.to(device).unsqueeze(0))
                mask = torch.argmax(prediction, dim=1).squeeze(0).cpu()
                mask = process_mask(image, mask)
                out = image_mask_overlay((x, mask))

                cv.imshow("Prediction", out)
                cv.waitKey(0)
                cv.destroyAllWindows()
        except KeyboardInterrupt:
            break
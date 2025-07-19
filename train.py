import argparse
import sys
import torch
from tqdm import tqdm

def train(loader, model, loss_fn, optimizer, device):
    model.train()
    sum_loss = 0.0
    for (X, y) in tqdm(loader):
        X, y = X.to(device), y.to(device)
        prediction = model(X)
        loss = loss_fn(prediction, y)

        # Backpropogation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        sum_loss += loss.item() * X.size(0)
    sum_loss /= len(loader.dataset)
    return sum_loss

def test(loader, model, loss_fn, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in tqdm(loader):
            X, y = X.to(device), y.to(device)
            prediction = model(X)
            test_loss += loss_fn(prediction, y).item() * X.size(0)
    test_loss /= len(loader.dataset)
    return test_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", help="model architecture identifier")
    parser.add_argument("--batch", type=int, help="batch size of initial training (default: 8)")
    parser.add_argument("--lr", type=float, help="initial training learning rate (default: 0.001)")
    parser.add_argument("--coeffs", help="initial training loss function coefficients for cross-entropy, Dice, Huber, and Focal-Tversky loss respectively (default: [0.25, 0.25, 0.25, 0.25])")
    parser.add_argument("--patience", type=int, help="initial training patience")
    parser.add_argument("--no_pretrain", action="store_true", help="disable ImageNet pre-training of model")
    parser.add_argument("--no_ft", action="store_true", help="disables finetuning of model with P2ILF")
    parser.add_argument("--batch_ft", type=int, help="batch size for finetuning (default: 8)")
    parser.add_argument("--lr_ft", type=float, help="finetuning learning rate (default: 0.0001)")
    args = parser.parse_args()

    possible_arch = ["unet", "unetpp", "unet3p", "resunetpp", "deeplabv3p"]
    if not args.arch:
        print("No model architecture given, please refer to README.")
        sys.exit(1)
    elif args.arch not in possible_arch:
        print("Unknown model architecture given, please refer to README.")
        sys.exit(1)
    
    if not args.batch:
        args.batch = 8
    
    if not args.lr:
        args.lr = 0.001
    
    if not args.coeffs:
        args.coeffs = [0.25, 0.25, 0.25, 0.25]
    else:
        if args.coeffs[0] != '[' or args.coeffs[-1] != ']':
            print("Invalid coefficient form, please write as a pythonic list (e.g. [0.25, 0.25, 0.25, 0.25])")
            sys.exit(1)
        coeffs = args.coeffs[1:-1]
        coeffs = coeffs.split(", ")
        for i in range(len(coeffs)):
            try:
                coeffs[i] = float(coeffs[i])
            except ValueError:
                print("Failed to convert loss function coefficient to float.")
                sys.exit(1)
        args.coeffs = coeffs

    if not args.patience:
        args.patience = 7

    if not args.batch_ft:
        args.batch_ft = 8
    
    if not args.lr_ft:
        args.lr_ft = 0.001
    
    print("Initialising...")

    import os

    from torchvision.transforms import Compose
    from torch.utils.data import DataLoader
    from torch.nn import CrossEntropyLoss, HuberLoss
    from segmentation_models_pytorch import Unet, UnetPlusPlus, DeepLabV3Plus
    from segmentation_models_pytorch.losses import DiceLoss, TverskyLoss
    from pathlib import Path

    from utils.dataset import L3D, P2ILF
    from utils.transform import AugmentIT, Resize, ToTensor, AugmentFT
    from utils.loss import ComboLoss
    from utils.class_weights import slippa_class_weights
    from models.unet3p.model import Unet3Plus
    from models.resunetpp.model import ResUnetPlusPlus
    
    train_data = L3D(
        "train",
        Compose([
            AugmentIT(),
            Resize(320, 416),
            ToTensor()
        ])
    )

    test_data = L3D(
        "val",
        Compose([
            Resize(320, 416),
            ToTensor()
        ])
    )

    train_loader = DataLoader(train_data, args.batch)
    test_loader = DataLoader(test_data, args.batch)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.2, 2)

    patience_counter = 0
    epoch_counter = 1
    lowest_loss = 10000
    last_model = None

    print("Calculating class weights...")
    temp_set = L3D("train", None)
    loss_fn = ComboLoss([
        CrossEntropyLoss(slippa_class_weights(temp_set, 4, device)),
        HuberLoss(),
        DiceLoss("multiclass", 4, ignore_index=0),
        TverskyLoss("multiclass", 4, ignore_index=0, alpha=0.5, beta=0.5, gamma=1.5) # Gamma is the focal component 
        ], args.coeffs)

    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    Path.mkdir(dir_path / "output", exist_ok=True)

    try:
        print("Training...")
        while patience_counter < args.patience:
            loss = train(train_loader, model, loss_fn, optimizer, device)
            val_loss = test(test_loader, model, loss_fn, device)
            scheduler.step(val_loss)
            print(f"EPOCH #{epoch_counter}: TRAIN = {loss} / VAL = {val_loss}")
        
            if val_loss > lowest_loss:
                patience_counter += 1
            else:
                lowest_loss = val_loss
                patience_counter = 0
                last_model = model.state_dict()

        print("Patience reached: saving model...")
        torch.save(model.state_dict(), dir_path / "output" / f"{args.arch}_lr{args.lr}_batch{args.batch}.pth")
    except KeyboardInterrupt:
        if last_model is not None:
            choice = input("Save current model? [y/N]: ")
            if choice.lower() == 'y':
                torch.save(last_model, dir_path / "output" / f"{args.arch}_lr{args.lr}_batch{args.batch}.pth")
            sys.exit(0)
    
    if not args.no_ft:
        del optimizer

        ft_train_data = P2ILF(
            "train",
            Compose([
                AugmentFT(),
                Resize(320, 416),
                ToTensor()
            ])
        )

        ft_test_data = P2ILF(
            "val",
            Compose([
                Resize(320, 416),
                ToTensor()
            ])
        )

        ft_train_loader = DataLoader(ft_train_data, args.batch_ft)
        ft_test_loader = DataLoader(ft_test_data, args.batch_ft)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_ft)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.2, 2)

        patience_counter = 0
        epoch_counter = 1
        lowest_loss = 10000
        last_model = None

        print("Calculating class weights...")
        temp_set = P2ILF("train", None)
        loss_fn = ComboLoss([
            CrossEntropyLoss(slippa_class_weights(temp_set, 4, device)),
            TverskyLoss("multiclass", 4, ignore_index=0, alpha=0.5, beta=0.5, gamma=1.5) # Gamma is the focal component 
            ], [0.75, 0.25])
        
        try:
            print("Finetuning...")
            while patience_counter < args.patience + 3:
                loss = train(ft_train_loader, model, loss_fn, optimizer, device)
                val_loss = test(ft_test_loader, model, loss_fn, device)
                scheduler.step(val_loss)
                print(f"EPOCH #{epoch_counter}: TRAIN = {loss} / VAL = {val_loss}")
            
                if val_loss > lowest_loss:
                    patience_counter += 1
                else:
                    lowest_loss = val_loss
                    patience_counter = 0
                    last_model = model.state_dict()

            print("Patience reached: saving model...")
            torch.save(model.state_dict(), dir_path / "output" / f"ft_{args.arch}_lr{args.lr}_batch{args.batch}.pth")
        except KeyboardInterrupt:
            if last_model is not None:
                choice = input("Save current model? [y/N]: ")
                if choice.lower() == 'y':
                    torch.save(last_model, dir_path / "output" / f"ft_{args.arch}_lr{args.lr}_batch{args.batch}.pth")
                sys.exit(0)

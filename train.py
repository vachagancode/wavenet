import math
import torch
import torch.nn as nn
from tqdm import tqdm

from wavenet import create_wavenet
from dataset import create_dataloaders
from utils import create_optimizer_and_scheduler
from config import get_config

def train(config, device, pretrained=None):
    # Create the model
    model = create_wavenet(config, device)

    train_dataloader, valid_dataloader, test_dataloader = create_dataloaders()

    if pretrained is not None:
        data = torch.load(pretrained, weights_only=True, map_location=device)
        model.load_state_dict(data["model_state_dict"])

        optimizer.load_state_dict(data["optimizer_state_dict"])
        scheduler.load_state_dict(data["scheduler_state_dict"])
        previous_loss = data["train_loss"]
        start_epoch = data["epoch"]
        end_epoch = start_epoch + config["epochs"]

        optimizer, scheduler = create_optimizer_and_scheduler(model, train_dataloader, start_epoch, end_epoch, data["optimizer_state_dict"], data["scheduler_state_dict"])

    else:
        start_epoch = 0
        end_epoch = start_epoch + config["epochs"]
        previous_loss = float("inf")

        optimizer, scheduler = create_optimizer_and_scheduler(model, train_dataloader, start_epoch, end_epoch)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, end_epoch):

        batch_loader = tqdm(train_dataloader)
        epoch_loss = 0
        epoch_step = 0
        for batch in batch_loader:
            model.train()

            src, tgt = batch
            src, tgt = src.float().to(device), tgt.long().to(device)

            # Do the forward pass
            logits = model(src)


            # Calculate the loss
            # print(logits.permute(0, 2, 1).shape, tgt.shape)
            loss = loss_fn(logits.permute(0, 2, 1), tgt.squeeze(1))


            batch_loader.set_postfix({"Loss": loss.item()})

            # Optimizer zero grad
            optimizer.zero_grad()

            # Loss Backwards
            loss.backward()

            # Optimizer + LR_Scheduler step
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_step += 1

        epoch_loss /= epoch_step

        print("Epoch: {epoch} | Loss: {epoch_loss}")
        # Save the model
        if epoch_loss < previous_loss:
            torch.save(
                obj={
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": epoch_loss,
                    "epoch": epoch
                },
                f=f"./models/me{epoch}l{math.floor(epoch_loss)}.pth"
            )
    return model

if __name__ == "__main__":
    config = get_config()

    train(config, device=torch.device('cpu'))

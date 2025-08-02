import os
import math
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

from wavenet import create_wavenet
from dataset import create_dataloaders
from utils import create_optimizer_and_scheduler, calculate_accuracy, create_summary_writer
from config import get_config

def train(config, device):
    print("[INFO] Experiment started.")
    start_time = time.time()

    # Create the model
    for model_config in config:
        print("-----------------------------------------------------------------------")
        print(f"[INFO] Model name: {model_config['model_name']}.")

        # Create the writer
        writer = create_summary_writer(model_config["model_name"])

        model = create_wavenet(model_config, device)

        train_dataloader, validation_dataloader, _ = create_dataloaders(model_config["pconv_output"])

        start_epoch = 0
        end_epoch = start_epoch + model_config["epochs"]
        previous_loss = float("inf")

        optimizer, scheduler = create_optimizer_and_scheduler(model, train_dataloader, start_epoch, end_epoch)

        loss_fn = nn.CrossEntropyLoss()
            for batch in batch_loader:
                model.train()

                src, tgt = batch
                src, tgt = src.float().to(device), tgt.long().to(device)

                # Do the forward pass
                logits = model(src) # [batch_size]

                # Calculate the loss
                # print(logits.permute(0, 2, 1).shape, tgt.shape)
                loss = loss_fn(logits.permute(0, 2, 1), tgt.squeeze(1))
                accuracy = calculate_accuracy(logits, tgt.squeeze(1))

                writer.add_scalar("Train/Loss", loss.item(), global_step=epoch)
                writer.add_scalar("Train/Accuracy", accuracy, global_step=epoch)

                learning_rate = optimizer.param_groups[0]['lr']
                batch_loader.set_postfix({"Loss": loss.item(), "Accuracy": accuracy, "Learning Rate": learning_rate})

                # Optimizer zero grad
                optimizer.zero_grad()

                # Loss Backwards
                loss.backward()

                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer + LR_Scheduler step
                optimizer.step()

                epoch_accuracy += accuracy
                epoch_loss += loss.item()
                epoch_step += 1

            epoch_loss /= epoch_step
            epoch_accuracy /= epoch_step

            print(f"[INFO] Training Epoch: {epoch} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.4f}%. | Learning Rate: {learning_rate}")

            if epoch % 2 == 0:
                # Do the validation
                model.eval()
                with torch.inference_mode():
                    valid_batch_loader = tqdm(validation_dataloader)
                    epoch_valid_loss = 0
                    epoch_valid_accuracy = 0
                    epoch_valid_step = 0
                    for batch in valid_batch_loader:
                        src, tgt = batch
                        src, tgt = src.float().to(device), tgt.long().to(device)

                        # Do the forward pass
                        valid_logits = model(src)

                        # Calculate the loss and accuracy]
                        valid_loss = loss_fn(valid_logits.permute(0, 2, 1), tgt.squeeze(1))
                        valid_accuracy = calculate_accuracy(valid_logits, tgt.squeeze(1))

                        epoch_valid_step += 1
                        epoch_valid_loss += valid_loss.item()
                        epoch_valid_accuracy += valid_accuracy

                        writer.add_scalar("Validation/Loss", valid_loss.item(), global_step=epoch)
                        writer.add_scalar("Validaion/Accuracy", valid_accuracy, global_step=epoch)

                    epoch_valid_loss /= epoch_valid_step
                    epoch_valid_accuracy /= epoch_valid_step
                    scheduler.step(epoch_valid_loss)

                print(f"[INFO] Validation Epoch: {epoch} | Loss: {epoch_valid_loss} | Accuracy: {epoch_valid_accuracy}%.")

                # Save the model
                if epoch_valid_loss < previous_loss:
                    torch.save(
                        obj={
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "valid_loss": epoch_valid_loss,
                            "epoch": epoch
                        },
                        f=f"./{new_model_dir}/me{epoch}l{math.floor(epoch_loss*100)}.pth"
                    )
                    previous_loss = epoch_valid_loss
                else:
                    epochs_not_improved += 1
                    if epochs_not_improved > early_stopping_patience:
                        print(f"[INFO] Training stopped at epoch {epoch} because the model is unable to improve.")
                        print(f"[INFO] Starting the training of the other model.")
                        break


                # Clear cuda cache
                if device == torch.device("cuda"):
                    torch.cuda.empty_cache()

        # Save the final models
        torch.save(
            obj={
                "model_state_dict": model.state_dict(),
            },
            f=f"./models/{model_config['model_name']}_final.pth"
        )
        writer.close()

    # Get the end time
    end_time = time.time()

    print(f"[INFO] Experiment was successfully finished in {(end_time  - start_time):.3f} seconds.")

if __name__ == "__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    train(config, device)

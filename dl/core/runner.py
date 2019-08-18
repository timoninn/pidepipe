from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

class Runner():
    def __init__(
        self,
        device=None
    ):
        self.device = device
        self.best_valid_loss = float('inf')

    def train(
        self,
        model: nn.Module,
        criterion: nn.Module,
        metric: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: _LRScheduler,
        loaders: {str: DataLoader},
        log_dir: str,
        num_epochs: int
    ):
        model.to(self.device)




        for epoch in range(num_epochs):


            # One batch

            # Train phase
            print(f'{epoch}/{num_epochs} Epoch {epoch} (train)')

            model.train()
            train_loader = loaders['train']

            num_batches = len(train_loader)
            tk = tqdm(train_loader, total=num_batches)

            running_loss = 0.0
            for itr, batch in enumerate(tk):

                # Train phase one batch
                images, masks = batch

                optimizer.zero_grad()

                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = model(images)

                loss = criterion(outputs, masks)

                loss.backward()
                optimizer.step()


                running_loss += loss.item()

                tk.set_postfix({'loss': running_loss / (itr + 1)})

            # Train phase loss.
            epoch_loss = running_loss / num_batches
            print(f'Loss: {epoch_loss:.4}')



            state = {
                "epoch": epoch,
                "loss": epoch_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

            torch.save(state, log_dir + 'last.pth')







            # Valid pahse
            print(f'{epoch}/{num_epochs} Epoch {epoch} (valid)')
            with torch.no_grad():
                model.eval()
                valid_loader = loaders['valid']

                num_batches = len(valid_loader)
                tk = tqdm(valid_loader, total=num_batches)

                running_loss = 0.0

                for itr, batch in enumerate(tk):
                    # Valid phase one batch

                    images, masks = batch

                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    outputs = model(images)

                    loss = criterion(outputs, masks)

                    running_loss += loss.item()

            # Valid phase loss.
            epoch_loss = running_loss / num_batches
            print(f'Loss: {epoch_loss:.4}')

            if epoch_loss < self.best_valid_loss:
                print('New optimal found. Saving state')
                self.best_valid_loss = epoch_loss

                torch.save(state, log_dir + 'best.pth')







    def infer(
        self,
        model: nn.Module,
        loaders: {str: DataLoader},
        checkpoint_path: str,
        save_logits_path: str
    ):
        # Infer phase
        checkpoint = torch.load(checkpoint_path)
        model_state_dict = checkpoint['model_state_dict']

        model.load_state_dict(model_state_dict)
        model.to(self.device)

        with torch.no_grad():
            model.eval()
            infer_loader = loaders['infer']
            num_batches = len(infer_loader)

            tq = tqdm(infer_loader, total=num_batches)

            result = []
            for itr, batch in enumerate(tq):
                # Infer phase batch.
                images, _ = batch

                images = images.to(self.device)

                outputs = model(images)

                for arr in outputs.cpu().numpy():
                    result.extend(arr)

            np.save(save_logits_path + 'logits.npy', result)

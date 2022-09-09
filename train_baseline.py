"""train_baseline.py


"""
import os 
import time 
import argparse

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from dataloaders import get_resampling_dataloader, get_slice_dataloaders
from baselines.mlp import MLP
from sklearn.metrics import f1_score, recall_score, precision_score
from protopnet.metrics import torch_metrics


def train(epochs: int, 
         dl_train: DataLoader, 
         dl_valid: DataLoader, 
         checkpoint_interval: int, 
         model_save_loc: str, 
         accumulate_steps: int) -> None:


    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    model = MLP(224, 10, 1).to(device)
    opt = optim.Adam(model.parameters(), lr=3e-4)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("model has parameters:", num_params)
    print(torch.cuda.get_device_name())
    print(device)

    start = time.perf_counter()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    global_step = 0
    training_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_recalls, val_recalls = [], []
    train_precs, val_precs = [], []
    train_f1s, val_f1s = [], []


    for epoch in range(epochs): 

        print("epoch: %d / %d" % (epoch, epochs))

        model.train()
        for batch_idx, (x_b, y_b) in enumerate(dl_train):
            if (batch_idx % 100) == 0: 
                print(f'-- training batch {batch_idx} / {len(dl_train)}\n', flush=True)

            x_b = x_b.to(device)
            y_b = y_b.to(device)

            raw_pred = model(x_b.float())

            acc, f1, prec, recall = torch_metrics(torch.sigmoid(raw_pred), y_b)

            # y_b_np = y_b.cpu().numpy()
            # pred_np = raw_pred.detach().cpu().numpy()
                  
            # recall = recall_score(y_b_np, pred_np, average='binary')
            # precision = precision_score(y_b_np, pred_np, average='binary')
            # f1 = f1_score(y_b_np, pred_np, average='binary')
            
            # store results
            train_accs.append(acc.item())
            train_f1s.append(f1.item())
            train_precs.append(prec.item())
            train_recalls.append(recall.item())

            loss = loss_fn(raw_pred, y_b.unsqueeze(1).float())
            loss.backward()

            if (global_step + 1) % accumulate_steps == 0: 
                opt.step()
                opt.zero_grad()
                training_losses.append(loss.item())
                if (global_step % 50) == 0: 
                    print('training loss %.3f, acc %.3f' % (loss.item(), acc.item()))

            global_step += 1
        
        print('beginning validation')
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            valid_acc = 0.0
            valid_f1 = 0.0
            valid_prec = 0.0 
            valid_recall = 0.0

            for x_batch, y_batch in dl_valid:
                x_batch = x_batch.to(device)
                y_batch = y_batch.float().to(device)

                valid_pred = model(x_batch.float())
                valid_loss += loss_fn(valid_pred, y_batch.unsqueeze(1).float()).item()

                acc, f1, prec, recall = torch_metrics(torch.sigmoid(valid_pred), y_batch)

                valid_acc += acc.item()
                valid_f1 += f1.item()
                valid_prec += prec.item()
                valid_recall += recall.item()

            print("validation loss", valid_loss / len(dl_valid))
            val_losses.append(valid_loss/len(dl_valid))

            print("validation acc", valid_acc / len(dl_valid))
            val_accs.append(valid_acc/len(dl_valid))

            print("validation f1", valid_f1 / len(dl_valid))
            val_f1s.append(valid_f1/len(dl_valid))

            print("validation prec", valid_prec / len(dl_valid))
            val_precs.append(valid_prec/len(dl_valid))

            print("validation recall", valid_recall / len(dl_valid))
            val_recalls.append(valid_recall/len(dl_valid))

        # save the stuff so that we can visualize outside of tensorboard
        np.savez(os.path.join(model_save_loc, 'scalars.npz'), 
            training_losses=training_losses, val_losses=val_losses, 
            train_accs=train_accs, val_accs = val_accs, 
            train_recalls = train_recalls, val_recalls = val_recalls,
            train_precs = train_precs, val_precs = val_precs,
            train_f1s = train_f1s, val_f1s = val_f1s)

        if epoch % checkpoint_interval == 0:
            print("checkpointing")
            torch.save({"epoch": epoch, 
                        "model_state_dict": model.state_dict(), 
                        "optimizer_state_dict": opt.state_dict()},
                os.path.join(model_save_loc, "checkpoint_%d.pth" % (epoch)))

        print("time elapsed: %d m, %.3f s" % ((time.perf_counter() - start) // 60, (time.perf_counter() - start) % 60))
        print("----")

    print('done training!!')
    return 


def main():

    args = parse_command_line()

    dl_train, dl_valid, dl_test = get_slice_dataloaders(args['data_dir'], 
                                                            {'train': args['metadata_train'], 
                                                            'val': args['metadata_valid'], 
                                                            'test': args['metadata_test']}, 
                                                            args['batch_sz'])
    
    # create folder for saving model training scalars, checkpoints, etc
    model_save_loc = os.path.join(args['model_save_dir'], args['trial_idx'])
    os.makedirs(model_save_loc, exist_ok=True)

    # train! 
    train(args['epochs'], 
         dl_train, dl_valid, 
         args['checkpoint_interval'],
         model_save_loc, 
         accumulate_steps=args['accumulate_grad']
    )

    return


def parse_command_line():
    parser = argparse.ArgumentParser()
    # get arguments for metadata, data location
    parser.add_argument('--metadata_train', action="store", type=str)
    parser.add_argument('--metadata_valid', action="store", type=str)
    parser.add_argument('--metadata_test', action="store", type=str)
    parser.add_argument('--data_dir', action="store", type=str)

    # get arguments for training hyperparam
    parser.add_argument('--batch_sz', action="store", type=int)
    parser.add_argument("--epochs", action="store", type=int)
    parser.add_argument("--checkpoint_interval", "--chkpt", action="store", type=int, default=5,)
    parser.add_argument("--accumulate_grad", action="store", type=int)

    # get arguments for output / intermediate storage
    parser.add_argument("--model_save_dir", action="store", type=str)
    parser.add_argument("--trial_idx", action="store", type=str)

    return vars(parser.parse_args())


if __name__ == "__main__":
    main()

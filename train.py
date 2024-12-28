import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import datetime
import matplotlib.pyplot as plt
import csv
import pandas as pd
import seaborn as sn

from vit_pytorch import ViT
from torch.utils.data import DataLoader
from Utils.getImage import getImage
from Models.CNN import CustomVGG16
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def main():
    # REQUIRED
    DATASET_ROOT = "../zdatasets/Pneu2 4.11.24/hasil_augmentasi/"

    # DONE = ['ChromaticAberration','CLAHE','ColorJitter','Defocus']
    # DONE = ['Emboss', 'Equalize']
    DONE = ['ZoomBlur']  
            
            
            # SKIP = 'ZoomBlur'

    header = ['Augmentation', 'loss_train', 'loss_valid', 'accuracy', 'f1 score', 'precision', 'recall', 'parameters',
              'flops', 'time per epoch','total_time']
    classes = ['PNEUMONIA', 'NORMAL']

    # print(len(DONE))
    # PARAMETER
    BATCH_SIZE = 32
    EPOCH = 500
    DEVICE = 'cuda'

    # HYPERPARAMETER
    LEARNING_RATE = 0.001
    flops = "15.3863G"

    for aug in DONE:  # os.listdir(DATASET_ROOT)
        start_all = datetime.datetime.now()
        start_load = datetime.datetime.now()
        # if aug in DONE:
        #     pass
        # else:
        train_loader = DataLoader(getImage(folder=DATASET_ROOT, augmentation=aug, state="TRAINING", pkl=False),
                                  batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(getImage(folder=DATASET_ROOT, augmentation=aug, state="VALIDATION", pkl=False),
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
        test_loader = DataLoader(getImage(folder=DATASET_ROOT, augmentation=aug, state="TESTING", pkl=False),
                                 batch_size=BATCH_SIZE, shuffle=True)
        #     DONE.append(aug)
        # print(DONE)
        end_load = datetime.datetime.now()
        print((end_load - start_load) / 60)

        model = CustomVGG16(dropout=0.5, device=DEVICE)
        # model = ViT(
        #     image_size=300,
        #     patch_size=10,
        #     num_classes=2,
        #     dim=1024,
        #     depth=1,
        #     heads=2,
        #     mlp_dim=1024,
        #     dropout=0.1,
        #     emb_dropout=0.1
        # ).to(DEVICE)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)  #

        loss_train, loss_valid, total_time = [], [], 0
        for epoch in range(EPOCH):
            start_epoch = datetime.datetime.now()
            loss_t, loss_v = 0, 0
            model.train()
            for batch, (src, trg) in enumerate(train_loader):
                src, trg = src.to(DEVICE), trg.to(DEVICE)
                src = torch.permute(src, (0, 3, 1, 2))

                pred = model(src).to(DEVICE)

                loss = loss_fn(trg, pred)
                loss_t += loss.cpu().detach().numpy()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            loss_train.append(loss_t / len(train_loader))

            model.eval()
            for batch, (src, trg) in enumerate(valid_loader):
                src, trg = src.to(DEVICE), trg.to(DEVICE)
                src = torch.permute(src, (0, 3, 1, 2))

                pred = model(src).to(DEVICE)

                loss = loss_fn(trg, pred)
                loss_v += loss.cpu().detach().numpy()
            loss_valid.append(loss_v / len(valid_loader))

            end_epoch = datetime.datetime.now()
            print("AUG = ", aug, ", EPOCH = ", epoch + 1, ", TRAINING LOSS = ", loss_t / len(train_loader),
                  ", VALID LOSS = ",
                  loss_v / len(valid_loader), ", time taken = ", end_epoch - start_epoch)

            f1 = plt.figure()
            plt.plot(range(len(loss_train)), loss_train, color="red")
            # plt.plot(range(len(loss_valid)), loss_valid, color="blue")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig("./evaldes5/Loss/training_" + aug +".png")
            plt.savefig("./evaldes5/Loss/training_" + aug + ".eps")

            if (epoch + 1) % 25 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_t / len(train_loader),
                }, "./Checkpoints/model_" + str(epoch + 1) + "_" + aug + "_" + str(LEARNING_RATE) + ".pt")

        prediction, ground_truth = [], []
        model.eval()
        for batch, (src, trg) in enumerate(test_loader):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            src = torch.permute(src, (0, 3, 1, 2))

            pred = model(src).to(DEVICE)

            prediction.extend(torch.argmax(pred, dim=1).cpu().detach().numpy())
            ground_truth.extend(torch.argmax(trg, dim=1).cpu().detach().numpy())

        cf_matrix = confusion_matrix(ground_truth, prediction)
        print(cf_matrix)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                             columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig("./evaldes5/Conf_Mat/" + aug + ".eps")
        plt.savefig("./evaldes5/Conf_Mat/" + aug + ".png")

        acc, pre, rec, f1 = accuracy_score(ground_truth, prediction), precision_score(ground_truth, prediction,
                                                                                      average='weighted'), recall_score(
            ground_truth, prediction, average='weighted'), f1_score(ground_truth, prediction, average='weighted')
        total_params = sum(p.numel() for p in model.parameters())

        end_all = datetime.datetime.now()
        with open('./REPORT_ALL_des5.csv', 'a') as f:
            writer = csv.writer(f)
            # writer.writerow(header)
            writer.writerow([aug, min(loss_train), min(loss_valid), acc, f1, pre, rec, total_params, flops,
                             end_epoch - start_epoch, end_all-start_all])


if __name__ == "__main__":
    main()

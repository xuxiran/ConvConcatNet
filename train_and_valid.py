# ------------------------Xiran Xu 2023-12-29----------------------------
# We use this code to train,valid and save the model.

allfiles = ['train', 'val', 'test']
# allfiles = ['test']


import glob
import config as cfg
from newmodel import Vlaai_sample
from EEG_Dataset import EEG_Dataset
from torch.utils.data import DataLoader
import tqdm
import torch
import os

# This is for ensemble model, you could change the seedstart to get different results using one GPU.
# So you could run "train_and_valid.py" for 100 times to get 100 results, and then run "ensemble_model.py" to get the final result.
# If you have many GPUs, you could run this code in each GPU by changing the seedstart, and then run "ensemble_model.py" to get the final result.
# For example, We have 20 GPUs, so we run this code in each GPU by changing the seedstart from 1*100 to 20*100, and then run "ensemble_model.py" to get the final result.
# We ensembled 520 models to get the final result.
seedstart = 0*100
seedend = seedstart + 100

for myseed in range(seedstart,seedend):
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(myseed)

    def correlation(y_pred, y_true):

        x = y_pred
        y = y_true

        xmean = torch.mean(x, 1, keepdim=True)
        ymean = torch.mean(y, 1, keepdim=True)
        vx = x - xmean
        vy = y - ymean
        cov = torch.sum(vx * vy, 1, keepdim=True)
        fenmu = (torch.sqrt(torch.sum(vx ** 2, 1, keepdim=True)) * torch.sqrt(
            torch.sum(vy ** 2, 1, keepdim=True)) + 1e-6)
        corr = cov / fenmu

        return corr


    # cross_validation
    valid_res = []
    valid_start_list = [0,26,48,71]
    valid_end_list = [26,48,71,85]
    for validcnt in range(4):
        print("validcnt:", validcnt)

        valid_start = valid_start_list[validcnt]
        valid_end = valid_end_list[validcnt]

        data_folder = cfg.data_folder
        # We use envelope to help the model to learn the stimulus features.
        stimulus_features = ["mel","envelope"]
        features = ["eeg"] + stimulus_features


        train_and_valid_files = [x for x in glob.glob(os.path.join(cfg.data_folder, "*")) if
                       os.path.basename(x).split("_")[0] in allfiles and
                       os.path.basename(x).split("_")[-1].split(".")[0] in features]

        valid_files_au = [x for x in train_and_valid_files if
                       valid_start < int(os.path.basename(x).split("-")[2][:-1]) <= valid_end]

        train_files = [x for x in train_and_valid_files if
                       x not in valid_files_au]

        # delete audiobook1 to avoid seen stimulus
        valid_files = [x for x in valid_files_au if
                       int(os.path.basename(x).split("_")[5]) != 1]
        valid_files_onlyau = [x for x in valid_files_au if
                       int(os.path.basename(x).split("_")[5]) == 1]


        train_dataset = EEG_Dataset(train_files)
        train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)
        valid_dataset = EEG_Dataset(valid_files)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfg.batch_size, shuffle=True)


        if cfg.model_name == 'Vlaai_sample':
            model = Vlaai_sample().to(cfg.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        valid_corr_sum = -1
        for epoch in range(cfg.epoch_num):

            # train the model
            num_correct = 0
            num_samples = 0
            train_loss = 0
            corr_sum = 0
            env_corr_sum = 0
            mel_corr_sum = 0

            # ---------------------train---------------------
            for iter, (eeg,env,mel) in enumerate(tqdm.tqdm(train_loader, position=0, leave=True), start=1):
                running_loss = 0.0
                # get the input
                eeg = eeg.to(cfg.device)
                mel = mel.to(cfg.device)
                env = env.to(cfg.device)
                mel  = torch.cat((env,mel),dim=2)

                # only average the mel when training
                mel_avg = mel.mean(dim=1, keepdim=True)
                mel_std = mel.std(dim=1, keepdim=True)
                mel = (mel - mel_avg) / mel_std

                pred = model(eeg)

                tmp_corr = correlation(pred,mel)
                env_corr = tmp_corr[:,:,0].mean()
                mel_corr = tmp_corr[:,:,1:].mean()

                loss = 1 - tmp_corr.mean()
                train_loss += loss

                corr_sum += tmp_corr.mean()
                env_corr_sum += env_corr
                mel_corr_sum += mel_corr

                # backward
                optimizer.zero_grad()  # clear the grad
                loss.backward()

                # gradient descent or adam step
                optimizer.step()

            print('epoch:', epoch)
            print('train_loss:', train_loss / iter)
            print('env_corr_sum:', env_corr_sum / iter)
            print('mel_corr_sum:', mel_corr_sum / iter)

            valid_loss = 0
            corr_sum = 0
            env_corr_sum = 0
            mel_corr_sum = 0

            # ---------------------valid---------------------
            model.eval()
            for iter, (eeg,env,mel) in enumerate(tqdm.tqdm(valid_loader, position=0, leave=True), start=1):
                with torch.no_grad():
                    running_loss = 0.0
                    # get the input
                    eeg = eeg.to(cfg.device)
                    env = env.to(cfg.device)
                    mel = mel.to(cfg.device) # when validing, we do not average the mel
                    mel  = torch.cat((env,mel),dim=2)


                    pred = model(eeg)

                    tmp_corr = correlation(pred, mel)
                    env_corr = tmp_corr[:,:,0]
                    mel_corr = tmp_corr[:,:,1:]
                    loss = 1 - tmp_corr.mean()

                    valid_loss += torch.mean(loss)
                    env_corr_sum += torch.mean(env_corr)
                    mel_corr_sum += torch.mean(mel_corr)

                    corr_sum += 1 - torch.mean(loss)

            # save the model
            if mel_corr_sum > valid_corr_sum:
                valid_corr_sum = mel_corr_sum

                saveckpt = './model/valid' + str(myseed) + '_' + str(valid_start) + '_' + str(valid_end) + '.ckpt'
                torch.save(model.state_dict(), saveckpt)

            print('epoch:', epoch)
            print('valid_loss:', valid_loss / iter)
            print('env_corr_sum:', env_corr_sum / iter)
            print('mel_corr_sum:', mel_corr_sum / iter)
            print('corr_sum:', corr_sum / iter)
            print('mel_corr_sum_max:', valid_corr_sum / iter)


        valid_res.append(valid_corr_sum / iter)

    print(valid_res)
    print(sum(valid_res)/len(valid_res))






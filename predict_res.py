# ------------------------Xiran Xu 2023-12-29----------------------------
# We use this code to get the prediction results.

import os
import glob
import json
import numpy as np
from newmodel import Vlaai_sample
import torch
import config as cfg


if __name__ == '__main__':

    # cfg.device = 'cpu'
    # load all eeg file
    data_folder = '../pre_eeg_mappings'
    eeg_folder = '../pre_eeg_mappings/preprocessed_eeg/'
    test_eeg_mapping = glob.glob(os.path.join(data_folder, 'sub*mapping.json'))




    for sub_stimulus_mapping_name in test_eeg_mapping:

        subject = os.path.basename(sub_stimulus_mapping_name).split('_')[0]
        print(subject)
        # load stimulus mapping
        sub_stimulus_mapping = json.load(open(sub_stimulus_mapping_name))

        #load eeg data
        sub_path = os.path.join(eeg_folder, f'{subject}_eeg.npz')
        sub_eeg_data = dict(np.load(sub_path))

        data_eeg =  np.stack([sub_eeg_data[value['eeg']]  for key, value in sub_stimulus_mapping.items() ])
        data_eeg_mean = np.mean(data_eeg, axis=1, keepdims=True)
        data_eeg_std = np.std(data_eeg, axis=1, keepdims=True)
        data_eeg = (data_eeg - data_eeg_mean) / data_eeg_std
        id_list= list(sub_stimulus_mapping.keys())

        data_eeg = torch.from_numpy(data_eeg).float().to(cfg.device)

        predict_list = []

        # load all models in the model folder
        models = []
        model_folder = './model'
        for model_name in os.listdir(model_folder):
            end_name = '.ckpt'
            if model_name.endswith(end_name):
                model = Vlaai_sample().to(cfg.device)
                model.load_state_dict(torch.load(os.path.join(model_folder, model_name)))
                # freeze the model
                for param in model.parameters():
                    param.requires_grad = False
                models.append(model)

        # predict
        outputs = [model(data_eeg) for model in models]
        outputs = torch.stack(outputs, dim=1)
        outputs_avg = outputs.mean(dim=2, keepdim=True)
        outputs_std = outputs.std(dim=2, keepdim=True)
        outputs = (outputs - outputs_avg) / outputs_std
        predictions = outputs.mean(dim=1)
        predictions = predictions[:, :, 1:]
        predictions = predictions.detach().to('cpu').numpy()

        # Make predictions json-serializable
        predictions = [np.array(value).tolist() for value in np.squeeze(predictions)]

        # Create dictionary from id_list and predictions
        sub = dict(zip(id_list, predictions))
        prediction_dir = os.path.join(os.path.dirname(__file__), 'predictions')
        os.makedirs(prediction_dir, exist_ok=True)
        with open(os.path.join(prediction_dir, subject + '.json'), 'w') as f:
            json.dump(sub, f)




import torch
data_folder = '../split_data'

device_ids = 0
device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")
epoch_num = 15
batch_size = 4
eeglen = 30*64

lr=1e-3
weight_decay=0.01

model_names = ['Vlaai_sample']
model_name = model_names[0]




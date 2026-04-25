import torch
import torch.nn as nn

class Satt(nn.Module):
    def __init__(self):
        super(Satt, self).__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 64)
        self.relu = nn.LeakyReLU()

    def forward(self, E):

        M_s = self.linear2(self.relu(self.linear1(E)))


        return M_s

class Extractor(nn.Module):
    def __init__(self, input_channels=128):
        super(Extractor, self).__init__()
        self.convs1 = nn.Conv1d(192, 64, 1)
        self.convt1 = nn.Conv1d(256, 256, 8,groups=256)

        self.convs2 = nn.Conv1d(256, 64, 1)
        self.convt2 = nn.Conv1d(256, 256, 8,groups=256)

        self.convs3 = nn.Conv1d(256, 64, 1)
        self.convt3 = nn.Conv1d(256, 256, 8,groups=256)

        self.convs4 = nn.Conv1d(256, 64, 1)
        self.convt4 = nn.Conv1d(256, 256, 8,groups=256)

        self.conv5 = nn.Conv1d(256, 128, 8)

        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(128)
        self.norm3 = nn.LayerNorm(256)

        self.relu = nn.LeakyReLU()
        self.pad = nn.ZeroPad2d((0, 0, 0, 7))

    def forward(self, x):
        eeg = x

        x = x.permute(0, 2, 1)
        x = self.convs1(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.relu(x)
        x = torch.cat((eeg, x), dim=2)

        x = x.permute(0, 2, 1)
        x = self.convt1(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pad(x)

        x = x.permute(0, 2, 1)
        x = self.convs2(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.relu(x)
        x = torch.cat((eeg, x), dim=2)

        x = x.permute(0, 2, 1)
        x = self.convt2(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pad(x)

        x = x.permute(0, 2, 1)
        x = self.convs3(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.relu(x)
        x = torch.cat((eeg, x), dim=2)

        x = x.permute(0, 2, 1)
        x = self.convt3(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pad(x)

        x = x.permute(0, 2, 1)
        x = self.convs4(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.relu(x)
        x = torch.cat((eeg, x), dim=2)

        x = x.permute(0, 2, 1)
        x = self.convt4(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pad(x)

        x = x.permute(0, 2, 1)
        x = self.conv5(x)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pad(x)
        return x

class OutputContext(nn.Module):
    def __init__(self, input_channels=64):
        super(OutputContext, self).__init__()

        self.pad = nn.ZeroPad2d((0, 0, 39, 0))
        self.conv = nn.Conv1d(input_channels, 64, 40)
        self.norm = nn.LayerNorm(64)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.pad(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.relu(x)
        return x

class Vlaai_sample(nn.Module):
    def __init__(self, nb_blocks=6, input_channels=64, output_dim=1):
        super(Vlaai_sample, self).__init__()

        self.extractor = Extractor(input_channels=input_channels)
        self.satt = Satt()
        self.output_context = OutputContext()
        self.nb_blocks = nb_blocks
        self.linear_layer = nn.Linear(128, 64)
        self.fc = nn.Linear(64, 11)

    def forward(self, x):
        eeg = x
        eeg_out = torch.zeros_like(eeg)
        eeg_out_att = torch.zeros_like(eeg)
        for i in range(self.nb_blocks):

            eeg_out = torch.cat((eeg,eeg_out,eeg_out_att), dim=2)

            eeg_out = self.extractor(eeg_out)

            eeg_out = self.linear_layer(eeg_out)
            eeg_out = self.output_context(eeg_out)

            eeg_out_att = self.satt(eeg_out)
            eeg_out_att = eeg_out_att * eeg_out

        eeg_out = self.fc(eeg_out.squeeze(dim=2))

        return eeg_out

if __name__ == '__main__':
    model = Vlaai_sample()
    x = torch.rand(4, 1920, 64)
    y = model(x)
    print(model(x).shape)

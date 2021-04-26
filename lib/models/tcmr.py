import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F

from lib.core.config import BASE_DATA_DIR
from lib.models.spin import Regressor


class TemporalAttention(nn.Module):
    def __init__(self, attention_size, seq_len, non_linearity='tanh'):
        super(TemporalAttention, self).__init__()

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        self.fc = nn.Linear(attention_size, 256)
        self.relu = nn.ReLU()
        self.attention = nn.Sequential(
            nn.Linear(256 * seq_len, 256),
            activation,
            nn.Linear(256, 256),
            activation,
            nn.Linear(256, seq_len),
            activation
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch = x.shape[0]
        x = self.fc(x)
        x = x.view(batch, -1)

        scores = self.attention(x)
        scores = self.softmax(scores)

        return scores


class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            seq_len=16,
            hidden_size=2048
    ):
        super(TemporalEncoder, self).__init__()

        self.gru_cur = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=n_layers
        )
        self.gru_bef = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=False,
            num_layers=n_layers
        )
        self.gru_aft = nn.GRU(
            input_size=2048,
            hidden_size=hidden_size,
            bidirectional=False,
            num_layers=n_layers
        )
        self.mid_frame = int(seq_len/2)
        self.hidden_size = hidden_size

        self.linear_cur = nn.Linear(hidden_size * 2, 2048)
        self.linear_bef = nn.Linear(hidden_size, 2048)
        self.linear_aft = nn.Linear(hidden_size, 2048)

        self.attention = TemporalAttention(attention_size=2048, seq_len=3, non_linearity='tanh')

    def forward(self, x, is_train=False):
        # NTF -> TNF
        y, state = self.gru_cur(x.permute(1,0,2))  # y: Tx N x (num_dirs x hidden size)

        x_bef = x[:, :self.mid_frame]
        x_aft = x[:, self.mid_frame+1:]
        x_aft = torch.flip(x_aft, dims=[1])
        y_bef, _ = self.gru_bef(x_bef.permute(1,0,2))
        y_aft, _ = self.gru_aft(x_aft.permute(1,0,2))

        # y_*: N x 2048
        y_cur = self.linear_cur(F.relu(y[self.mid_frame]))
        y_bef = self.linear_bef(F.relu(y_bef[-1]))
        y_aft = self.linear_aft(F.relu(y_aft[-1]))

        y = torch.cat((y_bef[:, None, :], y_cur[:, None, :], y_aft[:, None, :]), dim=1)

        scores = self.attention(y)
        out = torch.mul(y, scores[:, :, None])
        out = torch.sum(out, dim=1)  # N x 2048

        if not is_train:
            return out, scores
        else:
            y = torch.cat((y[:, 0:1], y[:, 2:], out[:, None, :]), dim=1)
            return y, scores


class TCMR(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            pretrained=osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(TCMR, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.encoder = \
            TemporalEncoder(
                seq_len=seqlen,
                n_layers=n_layers,
                hidden_size=hidden_size
            )

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')

    def forward(self, input, is_train=False, J_regressor=None):
        # input size NTF
        batch_size, seqlen = input.shape[:2]

        feature, scores = self.encoder(input, is_train=is_train)
        feature = feature.reshape(-1, feature.size(-1))

        smpl_output = self.regressor(feature, is_train=is_train, J_regressor=J_regressor)

        if not is_train:
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(batch_size, -1)
                s['verts'] = s['verts'].reshape(batch_size, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(batch_size, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(batch_size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(batch_size, -1, 3, 3)
                s['scores'] = scores

        else:
            repeat_num = 3
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(batch_size, repeat_num, -1)
                s['verts'] = s['verts'].reshape(batch_size, repeat_num, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(batch_size, repeat_num, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(batch_size, repeat_num, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(batch_size, repeat_num, -1, 3, 3)
                s['scores'] = scores

        return smpl_output, scores



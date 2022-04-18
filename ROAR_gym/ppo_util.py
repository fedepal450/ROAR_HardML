from pathlib import Path
from typing import Optional, Dict

import gym
import torch as th
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

cuda0 = th.device('cuda:0' if th.cuda.is_available() else 'cpu')


# model adapted from https://www.kaggle.com/mdteach/image-captioning-with-attention-pytorch/data
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()

        self.attention_dim = attention_dim

        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)

        self.A = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        u_hs = self.U(features)  # (batch_size,64,attention_dim)
        w_ah = self.W(hidden_state)  # (batch_size,attention_dim)

        combined_states = th.tanh(u_hs + w_ah.unsqueeze(1))  # (batch_size,64,attemtion_dim)

        attention_scores = self.A(combined_states)  # (batch_size,64,1)
        attention_scores = attention_scores.squeeze(2)  # (batch_size,64)

        alpha = F.softmax(attention_scores, dim=1)  # (batch_size,64)

        attention_weights = features * alpha.unsqueeze(2)  # (batch_size,64,features_dim)
        attention_weights = attention_weights.sum(dim=1)  # (batch_size,64)

        return attention_weights#,alphas


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, out_dim, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()

        # save the model param
        self.out_dim = out_dim
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim

        # self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        self.fcn = nn.Linear(decoder_dim, out_dim)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, features):
        # # vectorize the caption
        # embeds = self.embedding(captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        # get the seq length to iterate
        seq_length = features.size(0)
        batch_len = 1
        #num_features = features.size(1)

        outputs = th.zeros(batch_len, seq_length, self.out_dim).to(cuda0)

        for s in range(seq_length):
            context = self.attention(features, h)
            #lstm_input = th.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(context, (h, c))

            output = self.fcn(self.drop(h))
            # output =

            outputs[:, s] = output

        print(outputs.shape)

        return outputs

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c


class EncoderCNNtrain18(nn.Module):
    def __init__(self):
        super(EncoderCNNtrain18, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        # for param in resnet.parameters():
        #    param.requires_grad_(False)
        with th.no_grad():
            w = resnet.conv1.weight[:64,:2]

        modules = list(resnet.children())[:-1]
        modules[0] = th.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        modules[0].weight = nn.Parameter(w)
        #modules[-1] = th.flatten(, 1)


        self.resnet = nn.Sequential(*modules)

        self.lastResNet = list(resnet.children())[-1]

    def forward(self, images):
        features = self.resnet(images)  # (batch_size,512,8,8)
        features = th.flatten(features, 1)
        features = self.lastResNet(features)
        # features = features.permute(0, 2, 3, 1)  # (batch_size,8,8,512)
        # features = features.view(features.size(0), -1, features.size(-1))  # (batch_size,49,512)
        # print(features.shape)
        return features


class EncoderDecodertrain18(nn.Module):
    def __init__(self, embed_size, attention_dim, encoder_dim, decoder_dim, out_dim, drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNNtrain18()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            out_dim=out_dim
        )
        self.just_LSTM = th.nn.LSTM(input_size=1000,num_layers=1, hidden_size=256, batch_first=True)
        self.flat = nn.Sequential(
            th.nn.Flatten(start_dim=1, end_dim=-1),
            th.nn.Linear(25088, 512)
        )

    def forward(self, images):
        features = self.encoder(images) #features becomes 4,49,512
        # features = self.flat(features)
        features = th.reshape(features,(4,1,features.shape[1]))

        outputs = self.just_LSTM(features)

        out = outputs[0][0] #use for out shape [1,256]
        #out = th.reshape(outputs[0],(4,256))  # use for out shape [4,256]
        return out

class CustomMaxPoolCNN_attention(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomMaxPoolCNN_attention, self).__init__(observation_space, features_dim)
        # We assume CxWxH images (channels last)
        n_input_channels = observation_space.shape[0]

        self.fullStack = EncoderDecodertrain18(
            embed_size=200,
            attention_dim=300,
            encoder_dim=512,
            decoder_dim=300,
            out_dim=features_dim
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.fullStack(observations[0])



class CustomMaxPoolCNN(BaseFeaturesExtractor):
    """
    the CNN network that interleaves convolution & maxpooling layers, used in a
    previous DQN implementation and shows reasonable results
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomMaxPoolCNN, self).__init__(observation_space, features_dim)
        # We assume CxWxH images (channels last)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.Conv2d(6, 12, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(24, 48, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(48, 96, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(96, 96, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(96, 96, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=(4, 4)),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, n_flatten * 2), nn.ReLU(),
                                    nn.Linear(n_flatten * 2, n_flatten), nn.ReLU(),
                                    nn.Linear(n_flatten, features_dim), )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

v6_len = 12


class ResNet(nn.Module):
    def __init__(self, input_channels, inner_channels):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.Conv2d(input_channels, inner_channels, kernel_size=1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, input_channels, kernel_size=1),
        )

    def forward(self, inputs):
        return self.cnn(inputs) + inputs


class SingleFrame(nn.Module):
    def __init__(self, v6_len=13, rep_scale=20, k=8):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(2, k, kernel_size=(5, 5)),
            nn.ReLU(),
            ResNet(k, k * 4),
            ResNet(k, k * 4),
            nn.MaxPool2d(2),
            ResNet(k, k * 4),
            ResNet(k, k * 4),
            nn.MaxPool2d(2),
            ResNet(k, k * 4),
            ResNet(k, k * 4),
            nn.MaxPool2d(2),
            ResNet(k, k * 4),
            ResNet(k, k * 4),
            nn.BatchNorm2d(k),
            nn.Flatten()
        )
        self.Resnet18 = th.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # th.hub.load(resnet18)
        # resnet18 = models.resnet18()
        self.Resnet18.conv1 = th.nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Resnet18.fc = th.nn.Linear(512, 1)

        self.rep = nn.Linear(v6_len, v6_len * rep_scale)

    def forward(self, x, v6_len=13):
        frames = x[0:2]
        frames = frames.reshape(1,2,224,224)
        v6 = x[2]
        v6 = th.reshape(v6, (-1,))
        v6 = v6[0:v6_len]
        # print(frames.shape)
        one = self.Resnet18(frames)
        one = th.reshape(one, (-1,))
        two = self.rep(v6)
        out = th.cat((one, two))
        return out


class Part1(nn.Module):
    def __init__(self, v6_len=12):
        super().__init__()
        self.frame_process = SingleFrame()

    def forward(self, x, num_frames=4):
        head = []
        for i in range(num_frames):
            head.append(self.frame_process(x[0][i]))
            #1 fram_stack(batch)
        return th.stack(head)


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, input_dim)
        # batch_dim = number of samples per batch
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = th.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(cuda0)

        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        out, hn = self.rnn(x, h0)

        # Index hidden state of last time step
        # out.size() --> 100, 28, 10
        # out[:, -1, :] --> 100, 10 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


class Part2(nn.Module):
    def __init__(self, v6_len=6, input_dim=1260):
        super().__init__()
        self.seq = RNNModel(input_dim, input_dim * 2, 1, 256)

        # self.lstm = th.nn.LSTM(4, 1, input_size=input_dim)
        # (input_size=10, hidden_size=20, num_layers=2)

    def forward(self, x):
        x = x.reshape(1, len(x),len(x[0]))
        # print(x.shape)
        return self.seq(x)


class CustomMaxPoolCNN_combine(BaseFeaturesExtractor):
    """
    the CNN network that interleaves convolution & maxpooling layers, used in a
    previous DQN implementation and shows reasonable results
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomMaxPoolCNN_combine, self).__init__(observation_space, features_dim)
        # We assume CxWxH images (channels last)
        n_input_channels = observation_space.shape[0]

        self.fullStack = nn.Sequential(
            Part1(),
            Part2()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.fullStack(observations)

class CustomMaxPoolCNN_no_map(BaseFeaturesExtractor):
    """
    the CNN network that interleaves convolution & maxpooling layers, used in a
    previous DQN implementation and shows reasonable results
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomMaxPoolCNN_no_map, self).__init__(observation_space, features_dim)
        # We assume CxWxH images (channels last)
        n_input_channels = observation_space.shape[0]

        self.linear = nn.Sequential(nn.Linear(n_input_channels, 64), nn.ReLU(),
                                    nn.Linear(64, 128), nn.ReLU(),
                                    nn.Linear(128, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, features_dim), )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(observations)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

class Atari_PPO_Adapted_CNN(BaseFeaturesExtractor):
    """
    Based on https://github.com/DarylRodrigo/rl_lib/blob/master/PPO/Models.py
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(Atari_PPO_Adapted_CNN, self).__init__(observation_space,features_dim)
        channels = observation_space.shape[0]*observation_space.shape[1]
        self.network = nn.Sequential(
            # Scale(1/255),
            layer_init(nn.Conv2d(channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, features_dim)),
            # nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations=observations.view(observations.shape[0],-1,*observations.shape[3:])
        return self.network(observations)

def find_latest_model(root_path: Path) -> Optional[Path]:
    import os
    from pathlib import Path
    logs_path = (root_path / "logs")
    if logs_path.exists() is False:
        print(f"No previous record found in {logs_path}")
        return None
    paths = sorted(logs_path.iterdir(), key=os.path.getmtime)
    paths_dict: Dict[int, Path] = {
        int(path.name.split("_")[2]): path for path in paths
    }
    if len(paths_dict) == 0:
        return None
    latest_model_file_path: Optional[Path] = paths_dict[max(paths_dict.keys())]
    return latest_model_file_path

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv_block(in_ch, out_ch, kernel, stride=1, padding=2):
    return nn.Sequential(
        nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, padding=padding),
        nn.BatchNorm3d(num_features=out_ch),
        nn.PReLU()
    )


class ResBlock(nn.Module):
    def __init__(self, parameters):
        super(ResBlock, self).__init__()
        self.param = parameters
        self.conv_blocks = nn.Sequential(*[conv_block(in_ch=in_ch, out_ch=out_ch, kernel=self.param['kernel'])
                                           for in_ch, out_ch in
                                           zip(self.param['input_channels'], self.param['output_channels'])])

    def forward(self, x, input_from_skipped_connection=None):
        if input_from_skipped_connection is not None:
            x_and_skip = torch.cat((x, input_from_skipped_connection), dim=1)
            conv_output = self.conv_blocks(x_and_skip)
        else:
            conv_output = self.conv_blocks(x)

        try:
            res_output = conv_output + x.expand(-1, conv_output.shape[1], -1, -1, -1)
        except RuntimeError:
            res_output = conv_output + x_and_skip.expand(-1, conv_output.shape[1], -1, -1, -1)
        return res_output


class DownsamplingConv(nn.Module):

    def __init__(self, parameters):
        super(DownsamplingConv, self).__init__()
        self.param = parameters
        if 'res_block' in self.param.keys():
            self.residual_block = ResBlock(self.param['res_block'])
        else:
            self.residual_block = conv_block(self.param['conv_block']['input_channels'],
                                             self.param['conv_block']['output_channels'],
                                             self.param['conv_block']['kernel'])
        self.downsample = conv_block(self.param['downconv']['in_channels'],
                                     self.param['downconv']['out_channels'],
                                     self.param['downconv']['kernel'],
                                     self.param['downconv']['stride'],
                                     padding=0)

    def forward(self, x):
        res_output = self.residual_block(x)
        downsampled = self.downsample(res_output)
        return res_output, downsampled


class UpsamplingConv(nn.Module):

    def __init__(self, parameters):
        super(UpsamplingConv, self).__init__()
        self.param = parameters
        self.residual_block = ResBlock(self.param['res_block'])
        self.upsample = nn.Sequential(
            torch.nn.ConvTranspose3d(
                self.param['upconv']['in_channels'],
                self.param['upconv']['out_channels'],
                self.param['upconv']['kernel'],
                stride=2,
                padding=0
            ),
            nn.BatchNorm3d(num_features=self.param['upconv']['out_channels']),
            nn.PReLU()
        )

    def forward(self, x, input_skip):
        upsampled = self.upsample(x)
        res_output = self.residual_block(upsampled, input_skip)
        return res_output


class Encoder(nn.Module):
    def __init__(self, parameters):
        super(Encoder, self).__init__()
        self.param = parameters
        self.encode1 = DownsamplingConv(self.param['layer1'])
        self.encode2 = DownsamplingConv(self.param['layer2'])
        self.encode3 = DownsamplingConv(self.param['layer3'])
        self.encode4 = DownsamplingConv(self.param['layer4'])
        self.encode5 = ResBlock(self.param['layer5']['res_block'])
        self.key_encode = nn.Conv3d(in_channels=self.param['key_encoder']['in_channels'],
                                    out_channels=self.param['key_encoder']['out_channels'],
                                    kernel_size=self.param['key_encoder']['kernel'],
                                    padding=1)
        self.value_encode = nn.Conv3d(in_channels=self.param['value_encoder']['in_channels'],
                                      out_channels=self.param['value_encoder']['out_channels'],
                                      kernel_size=self.param['value_encoder']['kernel'],
                                      padding=1)

    def forward(self, input):
        skip1, out1 = self.encode1(input)
        skip2, out2 = self.encode2(out1)
        skip3, out3 = self.encode3(out2)
        skip4, out4 = self.encode4(out3)
        out5 = self.encode5(out4)
        key = self.key_encode(out5)
        value = self.value_encode(out5)
        return key, value, skip1, skip2, skip3, skip4


class Decoder(nn.Module):

    def __init__(self, parameters):
        super(Decoder, self).__init__()
        self.param = parameters

        self.decode1 = UpsamplingConv(self.param['layer1'])
        self.decode2 = UpsamplingConv(self.param['layer2'])
        self.decode3 = UpsamplingConv(self.param['layer3'])
        self.decode4 = UpsamplingConv(self.param['layer4'])
        self.decode5 = conv_block(self.param['layer5']['end_conv']['in_channels'],
                                  self.param['layer5']['end_conv']['out_channels'],
                                  self.param['layer5']['end_conv']['kernel'],
                                  padding=0)

        self.softmax = F.softmax

    def forward(self, x, skip1, skip2, skip3, skip4):
        out1 = self.decode1(x, skip4)
        out2 = self.decode2(out1, skip3)
        out3 = self.decode3(out2, skip2)
        out4 = self.decode4(out3, skip1)
        out = self.decode5(out4)
        # transform output to apply softmax voxel-wise
        batch, ch, h, w, d = out.shape
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        out = out.view(out.numel()//2, 2)
        out = self.softmax(out, dim=1)
        out = out.view(batch, h, w, d, ch)
        out = out.permute(0, 4, 1, 2, 3).contiguous()
        return out


class MemoryStorageAndReader:

    def __init__(self):
        self.keysM = None
        self.valuesM = None

    def record_to_memory(self, key, value):
        if self.keysM is not None:
            self.keysM = torch.cat((self.keysM, key), dim=0)
            self.valuesM = torch.cat((self.valuesM, value), dim=0)
        else:
            self.keysM = key
            self.valuesM = value

    def reset_memory(self):
        self.keysM = None
        self.valuesM = None

    def read(self, keyQ, valueQ):
        # T - time, C - channels, H - height, W - width, D - depth
        # TODO: implement computation for n-size minibatch
        T, C_keys, H, W, D = self.keysM.size()
        _, C_values, _, _, _ = self.valuesM.size()

        # Compare keys
        keyQ = keyQ.view(C_keys, H*W*D)
        keyQ = keyQ.permute(1, 0)
        keysM = self.keysM.permute(1, 0, 2, 3, 4)
        keysM = keysM.reshape(C_keys, T*H*W*D)
        key_similarity = torch.mm(keyQ, keysM)

        # Normalize and apply softmax
        key_similarity = key_similarity/math.sqrt(C_keys)
        key_similarity = F.softmax(key_similarity, dim=0)

        # Extract appropriate value from memory
        valueM = self.valuesM.permute(0, 2, 3, 4, 1)
        valueM = valueM.reshape(T*H*W*D, C_values)
        extracted_value = torch.mm(key_similarity, valueM)
        extracted_value = extracted_value.view(H, W, D, C_values)

        # Concatenate with query value
        read = torch.cat([valueQ, extracted_value.permute(3, 0, 1, 2).unsqueeze_(0)], dim=1)

        return read



class STNet(nn.Module):

    def __init__(self, parameters):
        super(STNet, self).__init__()
        self.param = parameters

        self.encoderM = Encoder(self.param['memory_encoder'])
        self.encoderQ = Encoder(self.param['query_encoder'])
        self.decoder = Decoder(self.param['decoder'])
        self.memoryUnit = MemoryStorageAndReader()


    def to_memory(self, x):
        for i in range(x.shape[1]):
            keyM, valueM, _, _, _, _ = self.encoderM(x[:, i, :, :, :, :])
            self.memoryUnit.record_to_memory(keyM, valueM)

    def segment(self, input):
        self.keyQ, self.valueQ, skip1, skip2, skip3, skip4 = self.encoderQ(input)
        read = self.memoryUnit.read(self.keyQ, self.valueQ)
        output = self.decoder(read, skip1, skip2, skip3, skip4)
        return output

    def forward(self, query, previous_data):
        torch.autograd.set_detect_anomaly(True)
        self.to_memory(previous_data)
        output = self.segment(query)
        return output








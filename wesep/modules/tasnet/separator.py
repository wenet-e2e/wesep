import torch.nn as nn

from wesep.modules.tasnet.convs import Conv1DBlock


class Separation(nn.Module):
    """
    R    Number of repeats
    X    Number of convolutional blocks in each repeat
    B    Number of channels in bottleneck and the residual paths
    H    Number of channels in convolutional blocks
    P    Kernel size in convolutional blocks
    norm The type of normalization(gln, cl, bn)
    causal  Two choice(causal or noncausal)
    skip_con Whether to use skip connection
    """

    def __init__(self, R, X, B, H, P, norm="gln", causal=False, skip_con=True):
        super(Separation, self).__init__()
        self.separation = nn.ModuleList([])
        for _ in range(R):
            for x in range(X):
                self.separation.append(
                    Conv1DBlock(B, H, P, 2**x, norm, causal, skip_con))
        self.skip_con = skip_con

    def forward(self, x):
        """
        x: [B, N, L]
        out: [B, N, L]
        """
        if self.skip_con:
            skip_connection = 0
            for i in range(len(self.separation)):
                skip, out = self.separation[i](x)
                skip_connection = skip_connection + skip
                x = out
            return skip_connection
        else:
            for i in range(len(self.separation)):
                out = self.separation[i](x)
                x = out
            return x

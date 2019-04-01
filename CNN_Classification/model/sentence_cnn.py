from torch import nn
from torch.nn import functional as F
from gluonnlp import Vocab

class SenCNN(nn.Module):
    def __init__(self, num_classes):
        super(SenCNN, self).__init__()
        
        # Static Embedding
        self.static_embedding = nn.Embedding.from_pretrained(torch.from_numpy(vocab.embedding.idx_to_vec.asnumpy()),
                                                    freeze=True)
        
        # Non-static Embedding
        self.non_static_embedding = nn.Embedding.from_pretrained(torch.from_numpy(vocab.embedding.idx_to_vec.asnumpy()),
                                                        freeze=False)

        # Conv Layers
        self.gram_3 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3)
        self.gram_4 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=4)
        self.gram_5 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=5)

        # Dropout
        self.dropout = nn.Dropout()

        # Output (fc)
        self.fc = nn.Linear(in_features=300, out_features=num_classes)


    def forward(self, x):
        static_embedding = self.static_embedding(x)
        non_static_embedding = self.non_static_embedding(x)

        # Feature extraction (conv)
        gram_3_feature = F.relu(self.gram_3(static_embedding)) + F.relu(self.gram_3(non_static_embedding))
        gram_4_feature = F.relu(self.gram_4(static_embedding)) + F.relu(self.gram_4(non_static_embedding))
        gram_5_feature = F.relu(self.gram_5(static_embedding)) + F.relu(self.gram_5(non_static_embedding))

        # max-over-time pooling
        
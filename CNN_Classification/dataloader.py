import os
from torch.utils.data import Dataset

class NaverSentimentDataset(Dataset):
    """Naver Sentiment Movie Corpus"""
    def __init__(self, txt_file: str, data_dir: str):
        """
        Args:
            txt_file (string): File path to txt file.
            root_dir (string): Directory to txt file.
        """
        
        nsmc_file = os.path.join(data_dir, txt_file)
        nsmc = open(nsmc_file, 'r', encoding='utf-8')
        nsmc_lines = nsmc.readlines()
        self.nsmc_txt = nsmc_lines[1:]
        nsmc.close()

    def __len__(self):
        return len(self.nsmc_txt)

    def __getitem__(self, idx):
        nsmc_single = self.nsmc_txt[idx]
        nsmc_split = nsmc_single.split('\t')
        
        nsmc_idx = nsmc_split[0].strip()
        nsmc_doc = nsmc_split[1].strip()
        nsmc_label = nsmc_split[2].strip()

        sample = {'id': nsmc_idx, 'text': nsmc_doc, 'label': nsmc_label}

        return sample

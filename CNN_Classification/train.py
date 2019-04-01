from torch.utils.data import DataLoader
from dataloader import NaverSentimentDataset


"""
for i in range(len(nsmc_dataset_train)):
    sample = nsmc_dataset_train[i]
    print("Current Index:", i)
    print("Input Text:", sample['text'])
    print("Label:", sample['label'])

    if i == 3:
        break
"""

# Helper function to show a batch
def nsmc_batch(sample_batch):
    """ Display text for a batch of samples """
    nsmc_text_batch, nsmc_label_batch = sample_batch['text'], sample_batch['label']
    batch_size = len(nsmc_text_batch)

    for i in range(batch_size):
        print("Text :", nsmc_text_batch[i])
        print("Labels :", nsmc_label_batch[i])
        print("Batch from dataloader")



if __name__ == "__main__":
    nsmc_dataset_train = NaverSentimentDataset(txt_file = 'ratings.txt', data_dir = 'data')
    dataloader = DataLoader(nsmc_dataset_train, batch_size=4, shuffle=True, num_workers=4)

    # Sample run to check dataloader
    for i_batch, sample_batch in enumerate(dataloader):
        print("Iter:", i_batch, "Batch Size:", len(sample_batch))

        # Observe 4th batch and stop
        if i_batch == 3:
            nsmc_batch(sample_batch)
            break

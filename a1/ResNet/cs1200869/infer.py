import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from resnet import *

class EvaluationDataset(Dataset):

    def __init__(self, filepath):
        self.images = torch.tensor(np.loadtxt(filepath, delimiter=',', dtype=np.float32))

        # image transformations
        mean = self.images.mean(axis=0)
        stdev = self.images.std(axis=0)
        
        self.images = (self.images - mean)/stdev
        
        self.images = self.images.reshape((self.images.shape[0],3,32,32))

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        return self.images[index]

if __name__ == "__main__":

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', required=True, type=str)
    parser.add_argument('--normalization', \
                        choices=['bn', 'in', 'bin', 'ln', 'gn', 'nn', 'inbuilt'])
    parser.add_argument('--n', type=int, choices=[1, 2, 3])
    parser.add_argument('--test_data_file', required=True, type=str)
    parser.add_argument('--output_file', required=True, type=str)

    args = parser.parse_args()

    model = torch.load(args.model_file, map_location=device)
    dataset = EvaluationDataset(args.test_data_file)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=False)

    all_preds = []
    for batch in dataloader:
        batch = batch.to(device)
        logits = model(batch)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds)

    all_preds = np.array(torch.hstack(all_preds).cpu())
    with open(args.output_file, 'w') as outfile:
        for val in all_preds:
            outfile.write(f"{val}\n")

from torch.utils.data.dataset import Dataset


class BaseDetectionDataset(Dataset):
    def __getitem__(self, item):
        raise NotImplementedError()

    def __get_datalist__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

from utils import train_load
from utils import dev_load
from utils import test_load
from torch.utils.data.dataset import Dataset
import torch
import numpy as np



class myDataset(Dataset):
    """
    Customized data set for loading train data, dev data and test data.
    Parameters:
    path: pathway to all the data files.
    parts: [1,2,3,4] or None.
    nframes: fixed number of frames in an utterance.
    mode: "train", "dev" or "test"
    embedding_flag: None, "enrol", "test" or "trail"
    """
    def __init__(self, path, parts, nframes, mode, embedding_flag):
        self.mode = mode
        self.nframes = nframes
        self.embedding_flag = embedding_flag
        # load data from preprocessed file
        if self.mode == "train":
            self.features, self.speakers, self.nspeakers = train_load(path, parts)
            print("features {}, speakers {}, nspeakers {}".format(
                self.features.shape, self.speakers.shape, self.nspeakers))
        elif self.mode == "dev":
            self.trials, self.labels, self.enrol, self.test = dev_load(path + "/dev.preprocessed.npz")
            print("trials {}, labels {}, enrol {}, test {}".format(
                self.trials.shape, self.labels.shape, self.enrol.shape, self.test.shape))
        else:
            self.trials, self.enrol, self.test = test_load(path + "/test.preprocessed.npz")
            print("trials {}, enrol {}, test {}".format(
                self.trials.shape, self.enrol.shape, self.test.shape))

    def __len__(self):
        if self.mode == "train":
            return self.features.shape[0]
        else: # if self.mode == "dev" or "test"
            if self.embedding_flag == "enrol":
                return self.enrol.shape[0]
            elif self.embedding_flag == "test":
                return self.test.shape[0]
            else: # if self.embedding_flag == "trail":
                return self.trials.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
            raw_inputs = np.copy(self.features[index])
            labels = self.speakers[index]
            inputs = trim_utterances(raw_inputs, self.nframes)
            inputs = inputs.reshape((1, inputs.shape[0], inputs.shape[1]))
            # if torch.cuda.is_available() is True:
                # return inputs, labels
            # else:
            return inputs.astype(float), labels

        else: # dev mode or test mode
            if self.embedding_flag == "enrol":
                raw_enrol = np.copy(self.enrol[index])
                enrol = trim_utterances(raw_enrol, self.nframes)
                enrol = enrol.reshape((1, enrol.shape[0], enrol.shape[1]))
                # if torch.cuda.is_available() is True:
                    # return enrol
                # else:
                return enrol.astype(float)

            elif self.embedding_flag == "test":
                raw_test = np.copy(self.test[index])
                test = trim_utterances(raw_test, self.nframes)
                test = test.reshape((1, test.shape[0], test.shape[1]))
                # if torch.cuda.is_available() is True:
                    # return test
                # else:
                return test.astype(float)

            else: # if self.embedding_flag == "trail"
                enrol_idx, test_idx = self.trials[index][0], self.trials[index][1]
                if self.mode == "dev":
                    label = self.labels[index]
                elif self.mode == "test":
                    label = False
                return enrol_idx, test_idx, str(label)


def trim_utterances(raw_inputs, nframes):
    if len(raw_inputs) < nframes:
        inputs = np.pad(raw_inputs, ((0,nframes-len(raw_inputs)), (0,0)), 'wrap')
    elif len(raw_inputs) > nframes:
        start = np.random.randint(0, high=len(raw_inputs)-nframes+1)
        inputs = np.copy(raw_inputs[start:start+nframes])
    else:
        inputs = raw_inputs
    return inputs


if __name__ == "__main__":
    dataset = myDataset("./data", [1], 9000, "train", None)
    print(dataset.__len__())
    for i in range(5):
        inputs, labels = dataset.__getitem__(i)
        print(inputs.shape, labels)
    dev_data = myDataset("./data", None, 9000, "dev", "enrol")
    print(dev_data.__len__())
    for i in range(5):
        enrol = dev_data.__getitem__(i)
        print(enrol.shape)
    dev_data.embedding_flag = "test"
    print(dev_data.__len__())
    for i in range(5):
        test = dev_data.__getitem__(i)
        print(test.shape)
    dev_data.embedding_flag = "trail"
    print(dev_data.__len__())
    for i in range(5):
        enrol_idx, test_idx, label = dev_data.__getitem__(i)
        print(enrol_idx, test_idx, label)



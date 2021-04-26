# EEGP
Easy-to-use EEG preprocessing pipeline based on [MNE][]

This project can help you build an EEG data  that is easy to share.

## Background
For beginners, processing EEG data is very complicated, and there are many to learn. At the same time, the pre-processing process is complicated, and even oneself is easy to forget details. This increases the difficulty for researchers to share scientific research results, and also makes it difficult to reproduce experimental results.

By using a unified EEG data preprocessing pipeline, it is convenient for researchers to share their research results, and experimental results are more easily reproduced. At the same time, beginners can quickly get started by reading the preprocessing code of others.

## Usage
You can refer to the implemented classes in paradigms folder to create preprocessing pipeline of your own.
When you have written your own class, others only need the following lines of code to reproduce your experimental results.

```python
from eegp.paradigms import MIFeetHand
from eegp.path import FilePath

class SaveData(MIFeetHand):
    def pipeline(self, filepaths):
        self.read_raw(filepaths)
        self.preprocess()
        self.make_epochs()
        self.make_data()
        self.save_data()

filepath = FilePath(subject='s1',
                     filetype='edf',
                     load_path='/tmp/data/s1.edf',
                     save_path='/tmp/data')
savedata = SaveData()
savedata.pipeline(filepaths)
```

## Demo
This is an [example][] that can be run online to let you know how easy it is to use when you share your pipeline with others.

## ToDo List



[MNE]:https://mne.tools/stable/index.html "MNE"
[example]:https://colab.research.google.com/drive/19hgfVbP47Ib-JUq4dyVxy77_INOAkSIG?usp=sharing "example"

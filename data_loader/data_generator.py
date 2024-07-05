import numpy as np
import pandas as pd
import random

class DataGenerator:
    def __init__(self,config):
        self.config = config
        # load data here
        self.train_data = pd.read_csv(r"D:\study\Tensorflow-Project-Template-master\data_loader\train.csv")
        self.test_data = pd.read_csv(r"D:\study\Tensorflow-Project-Template-master\data_loader\test.csv")
        self.train_features,self.train_labels,self.test_features = self.preprocessing(self.train_data,self.test_data)

    def preprocessing(self,train_data,test_data):

        all_features = pd.concat((train_data.iloc[:, 1:-1],
                                  test_data.iloc[:, 1:]))
        numeric_features = all_features.dtypes[all_features.dtypes !=
                                               'object'].index
        all_features[numeric_features] = all_features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std()))
        # 􀺽􀙵􀛸􀝸􀒅􀾯􀓻􁇙􀮄􁌱􀣐􊧊􀝒􀔅0􀒅􀲅􀕦􀝢􀕦􁍗􀴳􁊠0􀹶􀹊􀴘􁗌􀥦􊧊
        all_features = all_features.fillna(0)
        all_features = pd.get_dummies(all_features, dummy_na=True)
        n_train = train_data.shape[0]
        train_features = np.array(all_features[:n_train].values)
        test_features = np.array(all_features[n_train:].values)
        train_labels = np.array(train_data.SalePrice.values)
        train_labels = train_labels.reshape(-1,1)
        return train_features,train_labels,test_features

    def next_batch(self, batch_size):

        num_examples = len(self.train_features)
        indices = list(range(num_examples))
        random.shuffle(indices)
        for i in range(0, num_examples, batch_size):
            j = np.array(indices[i: min(i + batch_size, num_examples)])
            yield self.train_features[j], self.train_labels[j]

    def eval_data(self):
        return self.test_features



if __name__ == "__main__":
    data = DataGenerator(config=None)
    data_x,data_y = next(data.next_batch(10))
    print(data_x[0])



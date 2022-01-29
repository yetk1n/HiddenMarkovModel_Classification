from pomegranate import *
from hmmlearn import hmm
import pandas as pd
import numpy as np

train_path = ["train/2_audio.csv","train/3_audio.csv","train/4_audio.csv","train/5_audio.csv","train/6_audio.csv","train/7_audio.csv"
,"train/10_audio.csv","train/11_audio.csv","train/12_audio.csv","train/14_audio.csv","train/21_audio.csv","train/22_audio.csv",
"train/30_audio.csv","train/32_audio.csv","train/36_audio.csv","train/37_audio.csv","train/38_audio.csv","train/39_audio.csv"]
X_train = []
X_train_files_length = []
emission_failed = []
emission_notfailed = []
trans_mat_train = []
for file in train_path:
    Data = pd.read_csv(file, header=None)
    file_col = Data.T.values[2].astype(int)
    file_col = list(filter(lambda a: a != 0, file_col))
    X_train.append(np.array(file_col, dtype="int64"))
    for i in file_col:
        arr = []
        arr.append(i)
        trans_mat_train.append(arr)
    # Dividing failed and not failed files two part for emission probability
    if file == "train/21_audio.csv" or file == "train/22_audio.csv":
        for i in file_col:
            emission_notfailed.append(i)
    else:
        for i in file_col:
            emission_failed.append(i)

    
    X_train_files_length.append(len(file_col))

max_train_length = max(X_train_files_length)
emission_failed = np.array(emission_failed)
emission_notfailed = np.array(emission_notfailed)

X_train = np.array(X_train, dtype=object)

data = []
all_data = []
for i in X_train: 
    array = []
    for j in i:
        array.append(j)
    data.append(array)

# Observable calculation for start probability
start_failed = {}
start_notfailed = {}
label_train = pd.read_csv("labels/labels_train_resaved.csv")
count = 0
for i in label_train.T.values[1]:
    if i == 1:
        start_failed[label_train.T.values[0][count]] = i
    else:
        start_notfailed[label_train.T.values[0][count]] = i
    count += 1
startProbability = [len(start_notfailed)/(len(label_train.T.values[1])), len(start_failed)/(len(label_train.T.values[1]))]
startProbability = np.round(startProbability, 3)
print("\nStart Probability:")
print(startProbability)
print()

labels = []
for i in label_train.label.T.values:
    temp = []
    temp.append(i)
    labels.append(temp)


# Calculates the emission matrix manually
count_1F = np.count_nonzero(emission_failed == 1)
count_2F = np.count_nonzero(emission_failed == 2)
count_3F = np.count_nonzero(emission_failed == 3)
count_4F = np.count_nonzero(emission_failed == 4)
count_1N = np.count_nonzero(emission_notfailed == 1)
count_2N = np.count_nonzero(emission_notfailed == 2)
count_3N = np.count_nonzero(emission_notfailed == 3)
count_4N = np.count_nonzero(emission_notfailed == 4)
emissionProbability =[[count_1F/len(emission_failed), count_2F/len(emission_failed), count_3F/len(emission_failed), count_4F/len(emission_failed)],
                        [count_1N/len(emission_notfailed), count_2N/len(emission_notfailed), count_3N/len(emission_notfailed), count_4N/len(emission_notfailed)]]

emissionProbability = np.round(emissionProbability, 3)

print("Train Emission and Transition Probabilities:")
for i in emissionProbability:
    print(i)
model = hmm.MultinomialHMM(n_components=2, algorithm="viterbi", init_params="te")
model.startprob_ = startProbability
model.fit(trans_mat_train,lengths=X_train_files_length)
print()
print(model.transmat_)

print(50*"_")
print()

# Normalization for algorithm
p = 0
for i in data:
    for j in range(max_train_length - len(i)):
        data[p].append(None)
    p += 1
all_data.append(data)

# Training the model
model = HiddenMarkovModel.from_samples(PoissonDistribution, n_components = 2, X = all_data, algorithm = "viterbi", labels = labels)


test_path = ["test/15_audio.csv","test/16_audio.csv","test/17_audio.csv","test/20_audio.csv","test/25_audio.csv",
"test/28_audio.csv","test/40_audio.csv","test/42_audio.csv","test/43_audio.csv","test/44_audio.csv","test/45_audio.csv",
"test/46_audio.csv","test/48_audio.csv","test/49_audio.csv","test/50_audio.csv","test/52_audio.csv","test/53_audio.csv",
"test/54_audio.csv","test/55_audio.csv"]
X_test = []
X_test_files_length = []

for file in test_path:
    Data = pd.read_csv(file, header=None)
    file_col = Data.T.values[2].astype(int)
    file_col = list(filter(lambda a: a != 0, file_col))
    X_test.append(np.array(file_col, dtype="int64"))
    X_test_files_length.append(len(file_col))


max_test_length = max(X_test_files_length)
X_test = np.array(X_test, dtype = object)
data_test = []

for i in X_test:   
    array = []
    for j in i:
        array.append(j)
    data_test.append(array)

p = 0
for i in data_test:
    for j in range(max_test_length - len(i)):
        data_test[p].append(None)
    p += 1

# Predict test data
predict = model.predict(data_test, algorithm = "viterbi")
print("Results:")
for i in range(1,len(predict)):
    print(test_path[i-1], "   ", predict[i])

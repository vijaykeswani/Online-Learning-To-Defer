import numpy as np
from tqdm.notebook import tqdm
import random
from sklearn.neural_network import MLPClassifier
from collections import Counter

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import copy
from sklearn import tree
import sklearn


def flip(p):
    return 1 if random.random() < p else 0

def getAccuracy(predictions, labels):
    return np.mean(np.array(predictions) == labels)

def getAccuracyForGroup(predictions, labels, groups, g):
    predictions = [predictions[i]  for i in range(len(predictions)) if groups[i] == g]
    labels = [labels[i]  for i in range(len(labels)) if groups[i] == g]
    
    return np.mean(np.array(predictions) == labels)



class UniformExpert:
    def __init__(self, groups):
        self.groups = groups
    
    def prediction(self, label, group):
        pred = label if group in self.groups else flip(np.abs(label-0.8))
        return pred
    
    def predictionMod(self, label, group, sU, sL):
        pred = label if group in self.groups else flip(np.abs(label-0.8))
        score = sU if group in self.groups else sL
        pred = pred * score + (1-pred) * (1-score)
        return pred
    
    def dSim(self, group, sU, sL):
        score = sU if group in self.groups else sL
        return score
    

def getExpertCosts(num, costs=[]):
    if len(costs) == 0:
        c = 1
        costs = [c for _ in range(num)]
        costs.append(0)    
    else:
        costs = [2,1,0]
    return torch.Tensor(costs)

def getExpertPredictions(experts, label, group, sU, sL, dropout=0, modified=False):
    if not modified:
        expertPreds = [experts[j].prediction(label, group) for j in range(len(experts))]
    else:
        expertPreds = [experts[j].predictionMod(label, group, sU, sL) for j in range(len(experts))]
        
    expertPreds = [expertPreds[j] if flip(1-dropout) else 0 for j in range(len(experts))]    
    return expertPreds


def getSyntheticDataset(total=1000):
    X, y, groups = [], [], []
    group_frac = 0.5

    N1 = int(group_frac * total)
    d = 2

    N2 = int(N1/2)
    mu = np.array([random.random() for _ in range(d)])
    sig = [[0 for _ in range(d)] for _ in range(d)]
    for i in (range(d)):
        sig[i][i] = random.random()

    X = X + list(np.random.multivariate_normal(mu, sig, N2))
    X = X + list(np.random.multivariate_normal(mu+2.5, sig, N2))

    for _ in range(N2):
        groups.append(0)
        y.append(0)

    for _ in range(N2):
        groups.append(0)
        y.append(1)


    N3 = total - N1
    X = X + list(np.random.multivariate_normal(mu+5, sig, N3))

    for _ in range(N3):
        groups.append(1)

    for _ in range(N3):
        label = flip(0.5)
        y.append(label)

    return X, y, groups, mu, sig



def getPartition(frac=0.7, total=1000):
    indices = list(range(total))
    N = int(total * frac)
    
    random.shuffle(indices)
    train = indices[:N]
    test = indices[N:]
    
    return train, test


def getDeferrer_Syn(input_size, output_size):
    model = nn.Sequential(
        nn.Linear(input_size, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, output_size),
        nn.ReLU(),
    )
    
    return model

def getDeferrerWithPrior_Syn(input_size, output_size, X, groups, train, experts, sU, sL):
    model = nn.Sequential(
        nn.Linear(input_size, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, output_size),
    )
    
    
    while True:
        optimizer = optim.SGD(list(model.parameters()), lr=0.001)
        criterion = nn.MSELoss()
        for _ in tqdm(range(500)):

            random.shuffle(train)
            train_features = [X[i] for i in train[:32]]
            train_groups = [groups[i] for i in train[:32]]
            prob = [list([e.dSim(g, sU, sL) for e in experts]) + [0.1] for g in train_groups]

            loss = 0
            optimizer.zero_grad()
            for feat, p in zip(train_features, prob):
                output = model(torch.Tensor(feat))
                loss += criterion(output, torch.Tensor(p))
        #         print (loss)
            loss.backward()
            optimizer.step()
    #         print (loss.item(), output, torch.Tensor(p))

        outputs = [model(torch.Tensor(feat)).tolist() for feat in train_features]
        if np.abs(outputs[0][0] - prob[0][0]) < 0.05:
            break
        
#     print (train_groups[0], outputs[0], prob[0])
#     print (train_groups[1], outputs[1], prob[1])
    
    return model



def getClassifier_Syn(input_size):
    model = nn.Sequential(
        nn.Linear(input_size, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(4, 1),
    )

    return model

def getFinalPrediction(output, expertPreds):
    output = output.tolist()
    return np.dot(output, expertPreds) > 0.5

def getFinalPrediction2(output, expertPreds):
    output = output.tolist()
    output2 = np.dot(output, expertPreds)
    output2 = np.exp(output2)/(np.exp(1 - output2) + np.exp(output2))
    return output2 > 0.5

def getFinalPrediction3(output, expertPreds):
    output = output.tolist()
    return expertPreds[np.argmax(output)]


def train_preproc(deferrer_prior, experts, expert_costs, train, X, y, groups, lr=0.0075, ):
    deferrer = copy.deepcopy(deferrer_prior)
    classifier = sklearn.tree.DecisionTreeClassifier()

    df_losses, dfc_losses, clf_losses = [], [], []
    batch_size = 10
    xs = list(range(1, len(train)+batch_size, batch_size))

    crowdIndices, crowdFeatures, crowdLabels = [], {}, {}
    optimizer = optim.SGD(list(deferrer.parameters()), lr=lr)

    for i, t in enumerate(tqdm(xs)): # data-stream loop
        train_hist = list(train[:t])

        random.shuffle(train_hist)

        trainFeatures = [X[i] for i in train_hist]
        trainGroups = [int(groups[i]) for i in train_hist]
        trainLabels_true = [int(y[i]) for i in train_hist]


        outputs, inputs, expPreds = [], [], []
        loss_clf, loss_all, cost = 0, 0, 0
        optimizer.zero_grad()

        for index, feat, label_true, group in zip(train_hist, trainFeatures, trainLabels_true, trainGroups):

            if index in crowdIndices:
                label_final = int(crowdLabels[index] > 0.5)
                old = True

            else:

                x = torch.Tensor(feat)
                output = deferrer(x)
                output = torch.clamp(output, min=0, max=1)
                if i > 1:
                    clf_pred = classifier.predict([feat])[0]
                else:
                    clf_pred = 0

                expertPreds = getExpertPredictions(experts, label_true, group, 0, 1, 0, modified=False)
                expertPreds.append(clf_pred)
                expertPreds = torch.Tensor(expertPreds)

                label_crowd = torch.dot(output, expertPreds)
                label_crowd = torch.exp(label_crowd)/(torch.exp(1 - label_crowd) + torch.exp(label_crowd))
                label_final = int(label_crowd.item() > 0.5)

                crowdLabels[index] = label_crowd.item()
                crowdIndices.append(index)
                old = False

            if not old:
                loss_all -= torch.log(label_crowd) if label_final else torch.log(1-label_crowd)
                cost += torch.dot(torch.abs(output), expert_costs)


        lambda_cost = i/200
        if i % 2 == 0:
            loss = (loss_all + lambda_cost * cost)/batch_size
            loss.backward()
            optimizer.step()        
        else:
            classifier.fit(np.array([X[j] for j in crowdIndices]), np.array([int(crowdLabels[j] > 0.5) for j in crowdIndices]))

        df_losses.append(loss_all.item()/batch_size)
        dfc_losses.append((loss_all.item() + lambda_cost * cost.item())/batch_size)

    return deferrer, classifier

def test_syn(deferrer, classifier, experts, test, X, y, groups):
    testFeatures = [X[i] for i in test]
    testLabels = [int(y[i]) for i in test]
    testGroups = [int(groups[i]) for i in test]


    a = 0
    random_pred, random_fair_pred, joint_pred, joint_pred_fair, joint_pred_sp, joint_pred_fair_sp = [], [], [], [], [], []
    clf_pred = []
    wts, xs = [], []
    for feat, label, group in zip(testFeatures, testLabels, testGroups):
        x = torch.Tensor(feat)
        xs.append(feat[0])
        output = deferrer(x)
        output = torch.clamp(output, min=0, max=1)

        expertPreds = getExpertPredictions(experts, label, group, 0, 1, dropout=0)
        c_pred = classifier.predict([feat])[0]

        expertPreds.append(c_pred)
        clf_pred.append(c_pred)


        joint_pred.append(getFinalPrediction3(output, expertPreds))    

        wts.append([1 if j == np.max(output.tolist()) else 0 for j in output.tolist()])

    acc = (getAccuracy(joint_pred, testLabels))
#     print (acc, "\n")


    acc_0 = (getAccuracyForGroup(joint_pred, testLabels, testGroups, 0))
    acc_1 = (getAccuracyForGroup(joint_pred, testLabels, testGroups, 1))
#     print ("Joint pred", acc_0, acc_1)
#     accs.append([acc, acc_0, acc_1])

    c_acc = (getAccuracy(clf_pred, testLabels))
    c_acc_0 = (getAccuracyForGroup(clf_pred, testLabels, testGroups, 0))
    c_acc_1 = (getAccuracyForGroup(clf_pred, testLabels, testGroups, 1))
#     print ("Clf pred", acc_0, acc_1)
#     c_accs.append([acc, acc_0, acc_1])

    return ([acc, acc_0, acc_1], [c_acc, c_acc_0, c_acc_1], wts)



def train_smooth(deferrer, sL, sU, experts, expert_costs, train, X, y, groups, T=1000, lr=0.0075):
    classifier = sklearn.tree.DecisionTreeClassifier(max_depth=3)

    df_losses, dfc_losses, clf_losses = [], [], []
    batch_size = 5
    xs = list(range(1, len(train)+batch_size, batch_size))

    crowdIndices, crowdFeatures, crowdLabels = [], {}, {}
    optimizer = optim.SGD(list(deferrer.parameters()), lr=lr)

    for i2, t in enumerate(tqdm(xs)): # data-stream loop
        train_hist = list(train[:t])

        random.shuffle(train_hist)

        trainFeatures = [X[i] for i in train_hist]
        trainGroups = [int(groups[i]) for i in train_hist]
        trainLabels_true = [int(y[i]) for i in train_hist]

        mu = t/(t + T)
#             mu = 1/(1 + np.power(T/t, 2))
#             print (t, mu)

        outputs, inputs, expPreds = [], [], []
        loss_clf, loss_all, cost = 0, 0, 0
        optimizer.zero_grad()

        for index, feat, label_true, group in zip(train_hist, trainFeatures, trainLabels_true, trainGroups):

            if index in crowdIndices:
                label_final = int(crowdLabels[index] > 0.5)
                old = True

            else:

                x = torch.Tensor(feat)
                output = deferrer(x)
#                     output = torch.clamp(output, min=0, max=1)

                dSim = torch.Tensor(list([e.dSim(group, sU, sL) for e in experts]) + [0])
                output2 = mu * output + (1-mu) * dSim

                if i2 > 1:
                    clf_pred = classifier.predict([feat])[0]
                else:
                    clf_pred = 0

                expertPreds = getExpertPredictions(experts, label_true, group, sU, sL, 0, modified=False)
                expertPreds.append(clf_pred)
                expertPreds = torch.Tensor(expertPreds)

                label_crowd = torch.dot(output2, expertPreds)
                label_crowd = torch.exp(label_crowd)/(torch.exp(1 - label_crowd) + torch.exp(label_crowd))
                label_final = int(label_crowd.item() > 0.5)

                crowdLabels[index] = label_crowd.item()
                crowdIndices.append(index)
                old = False

            if not old:
                loss_all -= torch.log(label_crowd) if label_final else torch.log(1-label_crowd)
                cost += torch.dot(torch.abs(output2), expert_costs)

        lambda_cost = 0
        if i2 % 2 == 0:
            loss = (loss_all + lambda_cost * cost)/batch_size
            loss.backward()
            optimizer.step()        
        else:
            classifier.fit(np.array([X[j] for j in crowdIndices]), np.array([int(crowdLabels[j] > 0.5) for j in crowdIndices]))

        df_losses.append(loss_all.item()/batch_size)
        dfc_losses.append((loss_all.item() + lambda_cost * cost.item())/batch_size)
    
    return deferrer, classifier


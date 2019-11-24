import numpy as np
import random
import pandas as pd


def minkowskiDist(v1, v2, p):
    """
    Assumes v1 and v2 are equal-length lists of numbers
    Returns Minkowski distance of order p between v1 and v2
    """
    dist = 0.0
    for i in range(len(v1)):
        dist += abs(v1[i] - v2[i])**p
    return dist**(1/p)


class Patient(object):
    def __init__(self, features, label):
        """
        features a list of parameters of the patient
        label is the status of the patient
        """
        self.features = features
        self.label = label

    def getFeatures(self):
        return self.features

    def distance(self, other):
        """
        return the distance between one patient and another patient
        """
        return minkowskiDist(self.getFeatures(),
                             other.getFeatures(), 2)

    def getLabel(self):
        return self.label


def getOldData(fname, parameters):
    """
    fname: a string specifying the directory of the the file
    parameters: a list of strings containing the parameters of the patients
    the last element of the string is the status of the patient: whether sick
    or not/diagnosed or not
    return a dictionary of the data from different columns
    also return a list of status of each patient
    """
    data = {}
    df = pd.read_csv(fname)
    for i in range(len(parameters)):
        data[parameters[i]] = []
    for j in parameters:
        for k in range(len(df[j])):
            data[j].append(df[j][k])
    status = list(df["status"])
    return data, status


# # testing
# directory = "/Users/chunyuema/Desktop/CAREER/CS/ds/ds_project/parkinsons.csv"
# parameters = ["MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:Jitter(Abs)"]
# getOldData(directory, parameters)


def buildParkinsonPatients(fname, parameters):
    """
    fname: a string specifying the directory of the file
    return a list of Patients
    """
    data, status = getOldData(fname, parameters)
    patients = []
    for i in range(len(data[parameters[0]])):
        features = []
        for j in data.keys():
            features.append(data[j][i])
        # print(features)
        p = Patient(features, status[i])
        # print(p)
        patients.append(p)
    # print(patients)
    return patients


# # testing
# directory = "/Users/chunyuema/Desktop/CAREER/CS/ds/ds_project/parkinsons.csv"
# parameters = ["MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP"]
# Patients = buildParkinsonPatients(directory, parameters)


def findKNearest(patient, patientSet, k):
    """
    patient: a Patient object
    patientSet: a list of Patients
    k: an integer, the number of neearest patient object we are finding
    return:
    a list of Patients that are closest to the selected Patient
    a list of corresponding distances
    """
    kNearest, distances = [], []
    for i in range(k):
        kNearest.append(patientSet[i])
        distances.append(patient.distance(patientSet[i]))
    # print(kNearest)
    # print(distances)
    maxDist = max(distances)
    # print("max distance is: ", maxDist)
    for p in patientSet[k:]:
        dist = patient.distance(p)
        if dist < maxDist:
            maxIndex = distances.index(maxDist)
            kNearest[maxIndex] = p
            distances[maxIndex] = dist
            maxDist = max(distances)
            # print(kNearest)
            # print(distances)
    return kNearest, distances


# # testing
# directory = "/Users/chunyuema/Desktop/CAREER/CS/ds/ds_project/parkinsons.csv"
# parameters = ["MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP"]
# Patients = buildParkinsonPatients(directory, parameters)
# patient_test = Patients[0]
# findKNearest(patient_test, Patients, 5)


def split80_20(patients):
    """
    patients is the list of patient obejcts
    """
    sampleIndices = random.sample(range(len(patients)), len(patients)//5)
    trainingSet, testSet = [], []
    for i in range(len(patients)):
        if i in sampleIndices:
            testSet.append(patients[i])
        else:
            trainingSet.append(patients[i])
    # print(trainingSet)
    # print(testSet)
    return trainingSet, testSet


# # testing
# directory = "/Users/chunyuema/Desktop/CAREER/CS/ds/ds_project/parkinsons.csv"
# parameters = ["MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP"]
# Patients = buildParkinsonPatients(directory, parameters)
# split80_20(Patients)


def KNearestClassify(training, testSet, k):
    """
    training: list of patients to train the data
    test: list of patients to test the data
    label:
    k:
    """
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for testCase in testSet:
        kNearest, distances = findKNearest(testCase, training, k)
        numMatch = 0
        for i in range(len(kNearest)):
            if kNearest[i].getLabel() == 1:
                numMatch += 1
        if numMatch > k//2:
            if testCase.getLabel() == 1:
                truePos += 1
            else:
                falsePos += 1
        else:
            if testCase.getLabel() != 1:
                trueNeg += 1
            else:
                falseNeg += 1
    # print(truePos, falsePos, trueNeg, falseNeg)
    return truePos, falsePos, trueNeg, falseNeg


# # testing
# directory = "/Users/chunyuema/Desktop/CAREER/CS/ds/ds_project/parkinsons.csv"
# parameters = ["MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP"]
# Patients = buildParkinsonPatients(directory, parameters)
# print(len(Patients))
# train, test = split80_20(Patients)
# print(len(train))
# print(len(test))
# KNearestClassify(train, test, 5)


def accuracy(truePos, falsePos, trueNeg, falseNeg):
    numerator = truePos + trueNeg
    denominator = truePos + trueNeg + falsePos + falseNeg
    return numerator/denominator


def sensitivity(truePos, falseNeg):
    try:
        return truePos/(truePos + falseNeg)
    except ZeroDivisionError:
        return float('nan')


def specificity(trueNeg, falsePos):
    try:
        return trueNeg/(trueNeg + falsePos)
    except ZeroDivisionError:
        return float('nan')


def posPredVal(truePos, falsePos):
    try:
        return truePos/(truePos + falsePos)
    except ZeroDivisionError:
        return float('nan')


def negPredVal(trueNeg, falseNeg):
    try:
        return trueNeg/(trueNeg + falseNeg)
    except ZeroDivisionError:
        return float('nan')


def getStats(truePos, falsePos, trueNeg, falseNeg, toPrint=True):
    accur = accuracy(truePos, falsePos, trueNeg, falseNeg)
    sens = sensitivity(truePos, falseNeg)
    spec = specificity(trueNeg, falsePos)
    ppv = posPredVal(truePos, falsePos)
    if toPrint:
        print(' Accuracy =', round(accur, 3))
        print(' Sensitivity =', round(sens, 3))
        print(' Specificity =', round(spec, 3))
        print(' Pos. Pred. Val. =', round(ppv, 3))
    return (accur, sens, spec, ppv)


# testing
directory = "/Users/chunyuema/Desktop/CAREER/CS/ds/ds_project/parkinsons.csv"
parameters = ["MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
              "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)"]
Patients = buildParkinsonPatients(directory, parameters)
train, test = split80_20(Patients)
truePos, falsePos, trueNeg, falseNeg = KNearestClassify(train, test, 5)
getStats(truePos, falsePos, trueNeg, falseNeg, toPrint=True)

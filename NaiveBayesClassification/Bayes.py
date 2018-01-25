import numpy as np

class NaiveBayes:

    def composeList(self, dataSet):
        """
        Compose unrepeated vocabulary set from input.
        :param dataSet: input vocabulary set
        :return: unrepeated vocabulary list
        """
        vocabSet = set([])
        for document in dataSet:
            vocabSet = vocabSet | set(document) # unrepeated set

        return list(vocabSet)

    def binarize(self, vocabList, targetSet):
        """
        -- Set-of-words model --

        Binarize target set according to vocabulary list.
        :param vocabList: integrated vocabulary list.
        :param targetSet: target vocabulary list.
        :return: binarized vocabulary list
        """
        binarizedVec = [0] * len(vocabList)
        for word in targetSet:
            if word in vocabList:
                binarizedVec[vocabList.index(word)] = 1
            else:
                print "The word: %s is not in my Vocabulary!" % word
        return binarizedVec

    def count(self, vocabList, targetSet):
        """
        -- Bag-of-words model --

        Binarize target set according to vocabulary list.
        :param vocabList: integrated vocabulary list.
        :param targetSet: target vocabulary list.
        :return: binarized vocabulary list
        """
        binarizedVec = [0] * len(vocabList)
        for word in targetSet:
            if word in vocabList:
                binarizedVec[vocabList.index(word)] += 1
            else:
                print "The word: %s is not in my Vocabulary!" % word
        return binarizedVec

    def train(self, matrix, category):
        """
        According to category of each vocabulary vector, calculate probability
        of each word in one label.
        :return:
        """
        numWords       = len(matrix)       # 6
        numWordsPerVec = len(matrix[0])    # 32

        # Calculate probability of abusive sentence
        pAbusive = sum(category) / float(numWords)

        """
        Method 1
        """
        #p0Num = np.zeros(numWordsPerVec)
        #p1Num = np.zeros(numWordsPerVec)
        #p0Denum = 0.0
        #p1Denum = 0.0

        """
        Method 2
        """
        p0Num = np.ones(numWordsPerVec)
        p1Num = np.ones(numWordsPerVec)
        p0Denum = 2.0
        p1Denum = 2.0

        for i in range(numWords):     # 6
            if category[i] == 1:
                p1Num   += matrix[i]
                p1Denum += sum(matrix[i])
            else:
                p0Num   += matrix[i]
                p0Denum += sum(matrix[i])

        p1Vect = p1Num / p1Denum      # probability of each word in the list under the condition of label 1
        p0Vect = p0Num / p0Denum      # probability of each word in the list under the condition of label 0

        return p0Vect, p1Vect, pAbusive

    def classify(self, vec2Classify, p0Vec, p1Vec, pClass1):
        p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
        p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
        if p1 > p0:
            return 1
        else:
            return 0

import unittest
import logging
import logging.config
import Bayes
#import time
import numpy as np

class BayesUnitTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.DEBUG,
        #logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S')

    def setUp(self):
        self.naiveBayes = Bayes.NaiveBayes()
        posts, classes = self.loadDataSet()
        self.vocabList = self.naiveBayes.composeList(posts)
        logging.log(logging.INFO, "Vocabulary List: " + str(self.vocabList))

        #start = time.time()

        vecMatrix = []
        for post in posts:
            binarizedVocab = self.naiveBayes.binarize(self.vocabList, post)
            #logging.log(logging.DEBUG, "Post: " + str(post))
            logging.log(logging.DEBUG, "Binaried vector: " + str(binarizedVocab))
            vecMatrix.append(binarizedVocab)

        #stop = time.time()
        #logging.log(logging.INFO, "Consume %s seconds" % str(stop - start))

        self.p0, self.p1, self.pAbusive = self.naiveBayes.train(vecMatrix, classes)

        logging.log(logging.INFO, "P0: \n" + str(self.p0))
        logging.log(logging.INFO, "P1: \n" + str(self.p1))

    def test_classified1(self):
        testEntry = ['love', 'my', 'dalmation']

        thisDoc = np.array(self.naiveBayes.binarize(self.vocabList, testEntry))
        logging.log(logging.INFO, str(testEntry) + ' is classified as: ' + str(self.naiveBayes.classify(thisDoc,
                                                                               self.p0, self.p1, self.pAbusive)))

    def test_classified2(self):
        testEntry = ['help', 'garbage']

        thisDoc = np.array(self.naiveBayes.binarize(self.vocabList, testEntry))
        logging.log(logging.INFO, str(testEntry) + ' is classified as: ' + str(self.naiveBayes.classify(thisDoc,
                                                                               self.p0, self.p1, self.pAbusive)))
        pass

    def loadDataSet(self):
        postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],               # Unabusive 0          my
                      ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],            # Abusive   1          stupid him
                      ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],               # Unabusive 0          my     him
                      ['stop', 'posting', 'stupid', 'worthless', 'garbage'],                     # Abusive   1          stupid
                      ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],         # Unabusive 0          my     him
                      ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]                  # Abusive   1          stupid
        classVec = [0, 1, 0, 1, 0, 1]                                                            # Label them manually
        return postingList, classVec

if __name__ == '__main__':
    unittest.main()

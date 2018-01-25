
class TextUtil:
    def textParse(self, bigString):
        import re
        tokenList = re.split(r'\W*', bigString)

        noEmptyTokenList = []
        for token in tokenList:
            if len(token) > 0:
                noEmptyTokenList.append(token.lower())

    def spamTest(self):
        docList   = [];
        classList = [];
        fullText  = [];


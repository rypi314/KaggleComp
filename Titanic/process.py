# process.py

def hello(test):
    print('hellos: ', test)


def loadTitanic(fileString):
    dfSource = pd.read_csv(fileString)
    print(dfSource.count())
    print(dfSource.head())
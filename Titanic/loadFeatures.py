# process.py

from pandas import concat


def hello(test):
    print('hellos: ', test)


def loadTitanic(fileString):
    import pandas as pd
    dfSource = pd.read_csv(fileString)

    oneHotSex = pd.get_dummies(dfSource.Sex, prefix='Sex')
    oneHotCabin = pd.get_dummies(dfSource.Cabin, prefix='Cabin')
    oneHotEmbarked = pd.get_dummies(dfSource.Embarked, prefix='Embarked')

    dfOneHot = dfSource.join(oneHotSex)
    dfOneHot = dfOneHot.join(oneHotCabin)
    dfOneHot = dfOneHot.join(oneHotEmbarked)

    X = dfOneHot[['Pclass','Sex_female', 'Embarked_C', 'Embarked_S', 'Embarked_S', 'Fare']]
    X.to_csv('Featured_'+fileString)
    
    try:
        y = dfOneHot['Survived']
        y.to_csv('Label_'+fileString)
    except:
        print('Test datset. Missing Label data.')
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

    dfSource['Age'].fillna(dfSource['Age'].mode()[0], inplace=True)
    
    dfFeature = dfSource.join(oneHotSex)
    dfFeature = dfFeature.join(oneHotCabin)
    dfFeature = dfFeature.join(oneHotEmbarked)


    X = dfFeature[['Pclass','Sex_female', 'Embarked_C', 'Embarked_S', 'Embarked_S', 'Age']]
    X.to_csv('Featured_'+fileString)
    
    try:
        y = dfFeature['Survived']
        y.to_csv('Label_'+fileString)
    except:
        print('Test datset. Missing Label data.')
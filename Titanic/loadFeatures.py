# process.py

def maleChild(row):
    if row['Age'] < 10 and row['Sex'] == 'male':
        return 1
    return 0

def loadTitanic(fileString):
    import pandas as pd
    dfSource = pd.read_csv(fileString)

    oneHotSex = pd.get_dummies(dfSource.Sex, prefix='Sex')
    oneHotCabin = pd.get_dummies(dfSource.Cabin, prefix='Cabin')
    oneHotEmbarked = pd.get_dummies(dfSource.Embarked, prefix='Embarked')

    dfSource['Age'].fillna(dfSource['Age'].mode()[0], inplace=True)

    dfSource['MaleChild'] = dfSource.apply(lambda row: maleChild(row), axis=1)

    dfSource['TopDeck'] = dfSource.Cabin.str[0].isin(['A', 'B', 'C', 'D'])
    
    dfFeature = dfSource.join(oneHotSex)
    dfFeature = dfFeature.join(oneHotCabin)
    dfFeature = dfFeature.join(oneHotEmbarked)


    X = dfFeature[['Pclass', 'Age', 'Sex_female', 'MaleChild','TopDeck', 'SibSp']]
    X.to_csv('Featured_'+fileString)
    
    try:
        y = dfFeature['Survived']
        y.to_csv('Label_'+fileString)
    except:
        print('Test datset. Missing Label data.')
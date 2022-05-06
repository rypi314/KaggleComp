# process.py

def maleChild(row):
    if row['Age'] < 10 and row['Sex'] == 'male':
        return 1
    return 0

def notAlone(row):
    if row['SibSp'] > 0 or row['Parch'] > 0:
        return 1
    return 0

def loadTitanic(fileString):
    import pandas as pd
    dfSource = pd.read_csv(fileString)

    oneHotSex = pd.get_dummies(dfSource.Sex, prefix='Sex')
    oneHotCabin = pd.get_dummies(dfSource.Cabin, prefix='Cabin')
    oneHotEmbarked = pd.get_dummies(dfSource.Embarked, prefix='Embarked')

    dfSource['Age'].fillna(dfSource['Age'].mode()[0], inplace=True)

    dfSource['NotAlone'] = dfSource.apply(lambda row: notAlone(row), axis=1)

    dfSource['Family'] = dfSource['SibSp'] + dfSource['Parch']

    data = [dfSource]#, dfTitanicTest]
    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    for dataset in data:
        # extract titles
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        # replace titles with a more common title or as Rare
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        # convert titles into numbers
        dataset['Title'] = dataset['Title'].map(titles)
        # filling NaN with 0, to get safe
        dataset['Title'] = dataset['Title'].fillna(0)

    import re
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    data = [dfSource]#, test_df]

    for dataset in data:
        dataset['Cabin'] = dataset['Cabin'].fillna("U0")
        dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
        dataset['Deck'] = dataset['Deck'].map(deck)
        dataset['Deck'] = dataset['Deck'].fillna(0)
        dataset['Deck'] = dataset['Deck'].astype(int)

    dfFeature = dfSource.join(oneHotSex)
    dfFeature = dfFeature.join(oneHotCabin)
    dfFeature = dfFeature.join(oneHotEmbarked)


    X = dfFeature[['Pclass', 'Age', 'Sex_female', 'Deck', 'Title']]
    X.to_csv('Featured_'+fileString)
    
    try:
        y = dfFeature['Survived']
        y.to_csv('Label_'+fileString)
    except:
        print('Test datset. Missing Label data.')
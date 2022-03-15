# This is a sample Python script.

# Press Mayus+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from datetime import datetime
import getopt
import os
import sys
import numpy as np
import pandas as pd
from sklearn.utils import column_or_1d
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pickle

k=1
d=1
p='./'
f="train.csv"
oFile=""
m="uniform"
r=0
classifier = "TARGET"

def datetime_to_epoch(d):
    return datetime.datetime(d).strftime('%s')
    

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'o:k:d:r:m:p:f:h',['output=','k=','d=','path=','iFile','h'])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-o','--output'):
            oFile = arg
        elif opt == '-k':
            k = int(arg)
        elif opt ==  '-d':
            d = int(arg)
        elif opt in ('-p', '--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-h','--help'):
            print(' -o outputFile \n -k numberOfItems \n -d distanceParameter \n -p inputFilePath \n -f inputFileName \n ')
            exit(1)
        elif opt in ('-m'):
            m = arg
        elif opt in ('-r'):
            r = arg
        

    if p == './':
        iFile=p+str(f)
    else:
        iFile = p+"/" + str(f)

    def coerce_to_unicode(x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)

    ml_dataset = pd.read_csv(iFile)

    #print(ml_dataset.head(5))

    #ml_dataset = ml_dataset[
    #    ['Largo de sepalo', 'Ancho de sepalo', 'Largo de petalo', 'Ancho de petalo', 'Especie']]
    
    columns = list(ml_dataset.columns)
    ml_dataset = ml_dataset[columns]


    categorical_features = []
    
    #numerical_features = ['Largo de sepalo', 'Ancho de sepalo', 'Largo de petalo', 'Ancho de petalo']
    numerical_features = list(ml_dataset.columns)
    numerical_features.remove(classifier)
    
    text_features = []
    
    for feature in categorical_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in numerical_features:
        if ml_dataset[feature].dtype == np.dtype('M8[ns]') or ( 
                hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
            ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
        else:
            ml_dataset[feature] = ml_dataset[feature].astype('double')



    #target_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2} #Categorias en las que vamos a encasillar las instancias
    categories = list(ml_dataset[classifier].unique())
    target_map = { str(categories[i]) : i for i in range(0, len(categories))}
    n_cat = len(target_map)
    ml_dataset['__target__'] = ml_dataset[classifier].map(str).map(target_map)
    del ml_dataset[classifier] 

    #ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    print(f)
    print(ml_dataset.head(5))


    train, test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']])
    print(train.head(5))                                                                                         
    print(train['__target__'].value_counts())                                                                    
    print(test['__target__'].value_counts())

    drop_rows_when_missing = []
    #impute_when_missing = [{'feature': 'Largo de sepalo', 'impute_with': 'MEAN'},
    #                       {'feature': 'Ancho de sepalo', 'impute_with': 'MEAN'},
    #                       {'feature': 'Largo de petalo', 'impute_with': 'MEAN'},
    #                       {'feature': 'Ancho de petalo', 'impute_with': 'MEAN'}]
    column_vals = list(ml_dataset.columns)
    column_vals.remove('__target__')
    impute_when_missing = []
    for i in range(0, len(column_vals)):
        impute_when_missing.append({'feature': column_vals[i], 'impute_with' : 'MEAN'})

    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]

    for feature in impute_when_missing:
        if feature['impute_with'] == 'MEAN':
            v = train[feature['feature']].mean()
        elif feature['impute_with'] == 'MEDIAN':
            v = train[feature['feature']].median()
        elif feature['impute_with'] == 'CREATE_CATEGORY':
            v = 'NULL_CATEGORY'
        elif feature['impute_with'] == 'MODE':
            v = train[feature['feature']].value_counts().index[0]
        elif feature['impute_with'] == 'CONSTANT':
            v = feature['value']
        train[feature['feature']] = train[feature['feature']].fillna(v)
        test[feature['feature']] = test[feature['feature']].fillna(v)


    column_vals = list(ml_dataset.columns)
    column_vals.remove('__target__')
    rescale_features={}
    for i in range(0, len(column_vals)): 
        rescale_features.update({column_vals[i] : 'AVGSTD'})
    
    
    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'MINMAX':
            _min = train[feature_name].min()
            _max = train[feature_name].max()
            scale = _max - _min
            shift = _min
        else:
            shift = train[feature_name].mean()
            scale = train[feature_name].std()
        if scale == 0.:
            del train[feature_name]
            del test[feature_name]
        else:
            train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
            test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale


    trainX = train.drop('__target__', axis=1) #Eliminamos la columna con el atributo que clasifica a las instancias
    #trainY = train['__target__']

    testX = test.drop('__target__', axis=1)
    #testY = test['__target__']

    trainY = np.array(train['__target__'])
    testY = np.array(test['__target__'])

    undersample = RandomUnderSampler(sampling_strategy=0.5)#la mayoria va a estar representada el doble de veces

    trainXUnder,trainYUnder = undersample.fit_resample(trainX,trainY)
    testXUnder,testYUnder = undersample.fit_resample(testX, testY)

    clf = KNeighborsClassifier(n_neighbors=k,
                          weights=m,
                          algorithm='auto',
                          leaf_size=30,
                          p=d)

    clf.class_weight = "balanced"

    clf.fit(trainX, trainY)


# Build up our result dataset

# The model is now trained, we can apply it to our test set:

    predictions = clf.predict(testX)
    probas = clf.predict_proba(testX)

    predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    cols = [
        u'probability_of_value_%s' % label
        for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
    ]
    probabilities = pd.DataFrame(data=probas, index=testX.index, columns=cols)

# Build scored dataset
    results_test = testX.join(predictions, how='left')
    results_test = results_test.join(probabilities, how='left')
    results_test = results_test.join(test['__target__'], how='left')
    results_test = results_test.rename(columns= {'__target__': 'TARGET'})

    i=0
    for real,pred in zip(testY,predictions):
        print(real,pred)
        i+=1
        if i>5:
            break

    print(f1_score(testY, predictions, average='macro'))
    print(classification_report(testY,predictions))
    print(confusion_matrix(testY, predictions))
    
    if oFile != "":    
        f = open(oFile, mode='a')
        if (n_cat == 2):
            if os.path.getsize(oFile) == 0:
                f.write("k, p, m, f1_score, recall, precision\n")
            f.write("%s, %s, %s" %(str(k),str(d), m))
            f.write(", %s, %s, %s" %(str(f1_score(testY,predictions, average=None)), str(recall_score(testY,predictions, average=None)), str(precision_score(testY,predictions, average=None)))+ "\n")
        elif (n_cat > 2):
            if os.path.getsize(oFile) == 0:
                f.write("k, p, m, MACRO_f1_score, MICRO_f1_score, AVG_f1_score, AVG_recall, AVG_precision\n")
            f.write("%s, %s, %s" %(str(k),str(d), m))
            f.write(", %s, %s, %s, %s, %s" %(str(f1_score(testY,predictions, average='macro')), str(f1_score(testY,predictions, average='micro')), str(f1_score(testY,predictions, average='weighted')), str(recall_score(testY,predictions,average="macro")), str(precision_score(testY,predictions, average='macro')))+ "\n")
        f.close()
        
    if r == '1':
        model = "knn.sav"
        saved_model = pickle.dump(clf, open(model,'wb'))
        print('Modelo guardado correctamente')
    
print("bukatu da")
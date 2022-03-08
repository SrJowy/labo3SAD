from datetime import datetime
import getopt
from re import S
import sys
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import os

k=1
d=1
p='./'
f="iris.csv"
oFile=""
aut = 1
classifier = "Especie"

def datetime_to_epoch(d):
    return datetime.datetime(d).strftime('%s')

if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'o:k:d:p:f:h:c:a',['output=','k=','d=','path=','iFile','h','classifier'])
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
        elif opt in ('-c', '--classifier'):
            classifier = arg
        elif opt in ('-a'):
            aut = 1

    if p == './':
        iFile=p+str(f)
        print(str(iFile))
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
    
    if aut == 0:
        classifier = raw_input("Type the name of the classifying attribute\n--> ")
        classifier = classifier.replace('\r','')
    
        columns = [item for item in str(input("Select the columns you want to use.\n Type -1 if you want all of them. \n --> ")).split()]
    else:
        columns = ['-1']
    
    
    if columns[0] == '-1':
        columns = list(ml_dataset.columns)
        ml_dataset = ml_dataset[columns]
    else:
        ml_dataset = ml_dataset[columns]
    
    if aut == 0:  
        catFeatures = [item for item in str(input ("Select the categorical attributes.\n Type -1 if you want all of them or -2 if you want none. \n --> ")).split()]
        numFeatures = [item for item in str(input ("Select the numerical features.\n Type -1 if you want all of them or -2 if you want none.. \n --> ")).split()]
        textFeatures = [item for item in str(input ("Select the text features.\n Type -1 if you want all of them or -2 if you want none.. \n --> ")).split()]
    else:
        catFeatures = ['-2']
        numFeatures = ['-1']
        textFeatures = ['-2']
    
    if catFeatures[0] == '-1':
        catFeatures = list(ml_dataset.columns)
        catFeatures.remove(classifier)
    elif catFeatures[0] == '-2':
        catFeatures = []
    
    if numFeatures[0] == '-1':
        numFeatures = list(ml_dataset.columns)
        numFeatures.remove(classifier)
    elif numFeatures[0] == '-2':
        numFeatures = []
        
    if textFeatures[0] == '-1':
        textFeatures = list(ml_dataset.columns)
        textFeatures.remove(classifier)
    elif textFeatures[0] == '-2':
        textFeatures = []
        
    for feature in catFeatures:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode) #Actualizar el texto a unicode

    for feature in textFeatures:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode) #Actualizar el texto a unicode

    for feature in numFeatures: #M8[ns] --> fecha de 64 bits
        if ml_dataset[feature].dtype == np.dtype('M8[ns]') or ( #Si el tipo del atributo es 'M8[ns]'
                hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')): #o tiene un atributo llamado 'base' y ese atributo es de tipo 'M8[ns]'
            ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature]) #convertimos esa fecha a epoch
        else:
            ml_dataset[feature] = ml_dataset[feature].astype('double') #Cambiamos el tipo el del atributo a double
    
    if aut == 0:    
        categories_str = str (raw_input("Type all the categories of the dataset separeted by commas \n"))
        categories_str_r = categories_str.replace('\r','')
        categories = categories_str_r.split(",")
    else:
        categories = list(ml_dataset[classifier].unique())
    
    target_map = { categories[i] : i for i in range(0, len(categories))}
    ml_dataset['__target__'] = ml_dataset[classifier].map(str).map(target_map)
    del ml_dataset[classifier]
    
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    print(f)
    #print(ml_dataset.head(120))
    
    train, test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']]) #Elegimos la muestra para entrenar el modelo,
    print(train.head(5))                                                                                         #EL 20% sera para test, indice aleatorio de 42
    print(train['__target__'].value_counts())                                                                    #y en base al dataset obtenido antes
    print(test['__target__'].value_counts())
    
    if aut == 0:
        drop_rows_when_missing = [item for item in str(input ("Select the features you want to remove if rows are missing.\n Type -1 if you want all of them or -2 if you want none. \n --> ")).split()]
        impute_features = [item for item in str(input ("Select the features you want to impute if rows are missing.\n Type -1 if you want all of them or -2 if you want none.. \n --> ")).split()]
    else:
        drop_rows_when_missing = ['-2']
        impute_features = ['-1']
    
    if drop_rows_when_missing[0] == '-1':
        drop_rows_when_missing = list(ml_dataset.columns)
        drop_rows_when_missing.remove(classifier)
    elif drop_rows_when_missing[0] == '-2':
        drop_rows_when_missing = []
    
    if impute_features[0] == '-1':
        impute_features = list(ml_dataset.columns)
        impute_features.remove('__target__')
        if aut == 0:
            param = raw_input("Select the parameter you want the features to be imputed with (MEAN, MEDIAN, CREATE_CATEGORY, MODE, CONSTANT)\n -->")
            param = param.replace('\r','')
        else:
            param = 'MEAN'
    elif impute_features[0] == '-2':
        impute_features = []
    else:
        if aut == 0:
            param = raw_input("Select the parameter you want the features to be imputed with (MEAN, MEDIAN, CREATE_CATEGORY, MODE, CONSTANT)\n -->")
            param = param.replace('\r','')
        else:
            param = 'MEAN'
    
    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]
        
    for feature in impute_features:
        if param == 'MEAN':
            v = train[feature].mean()
        elif param == 'MEDIAN':
            v = train[feature].median()
        elif param == 'CREATE_CATEGORY':
            v = 'NULL_CATEGORY'
        elif param == 'MODE':
            v = train[feature].value_counts().index[0]
        elif param == 'CONSTANT':
            v = feature['value']
        train[feature] = train[feature].fillna(v)
        test[feature] = test[feature].fillna(v)
        
    if aut == 0:
        rescale_features = [item for item in str(input ("Select the features you want to rescale.\n Type -1 if you want all of them or -2 if you want none. \n --> ")).split()]
    else:
        rescale_features = ['-1']
        
    if rescale_features[0] == '-1':
        rescale_features = list(ml_dataset.columns)
        rescale_features.remove('__target__')
        if aut == 0:
            param = raw_input("Select the parameter you want the features to be rescaled with (MINMAX, AVGSTD)\n -->")
            param = param.replace('\r','')
        else:
            param = 'AVGSTD'
            
    elif rescale_features[0] == '-2':
        rescale_features = []
    else:
        if aut == 0:
            param = raw_input("Select the parameter you want the features to be rescaled with (MINMAX, AVGSTD)\n -->")
            param = param.replace('\r','')
        else:
            param = 'AVGSTD'
        
    for feature in rescale_features:
        if param == 'MINMAX':
            _min = train[feature].min()
            _max = train[feature].max()
            scale = _max - _min
            shift = _min
        else:
            shift = train[feature].mean()
            scale = train[feature].std()
        if scale == 0.:
            del train[feature]
            del test[feature]
        else:
            train[feature] = (train[feature] - shift).astype(np.float64) / scale
            test[feature] = (test[feature] - shift).astype(np.float64) / scale
            
    
    trainX = train.drop('__target__', axis=1) #Eliminamos la columna con el atributo que clasifica a las instancias
    #trainY = train['__target__']

    testX = test.drop('__target__', axis=1)
    #testY = test['__target__']

    trainY = np.array(train['__target__'])
    testY = np.array(test['__target__'])

    # Balancear los datos en caso de que esten desbalanceados
    #undersample = RandomUnderSampler(sampling_strategy=0.5)#la mayoria va a estar representada el doble de veces

    #trainXUnder,trainYUnder = undersample.fit_resample(trainX,trainY)
    #testXUnder,testYUnder = undersample.fit_resample(testX, testY)

    # Calcular el valor del knn
    clf = KNeighborsClassifier(n_neighbors=k,
                          weights='uniform',
                          algorithm='auto',
                          leaf_size=30,
                          p=d)
    
    #k tendra que ser impar sino podria haber empates

    # Ponemos a cada clase un peso balanceado
    clf.class_weight = "balanced"

    # Introducimos los valores para el entrenamiento

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

    print(f1_score(testY, predictions, average=None))
    print(classification_report(testY,predictions))
    print(confusion_matrix(testY, predictions))
    
    if oFile != "":    
        f = open(oFile, mode='a')
        if os.path.getsize(oFile) == 0:
            f.write("k, p, f1_score, recall, precision\n")
        f.write("%s, %s" %(str(k),str(d)))
        f.write(", %s, %s, %s" %(str(f1_score(testY,predictions, average='macro')), str(recall_score(testY,predictions,average="macro")), str(precision_score(testY,predictions, average='macro')))+ "\n")
        f.close()
    
print("fin")
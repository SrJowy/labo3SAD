# This is a sample Python script.

# Press Mayus+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import getopt
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

k=1
#kM=sys.argv[2]
d=2

p='./'
f="iris.csv"
oFile="datos.csv"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'o:k:d:p:f:h',['output=','k=','d=','path=','iFile','h'])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-o','--output'):
            oFile = arg
        elif opt == '-k':
            k = arg
        elif opt ==  '-d':
            d = arg
        elif opt in ('-p', '--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-h','--help'):
            print(' -o outputFile \n -k numberOfItems \n -d distanceParameter \n -p inputFilePath \n -f inputFileName \n ')
            exit(1)

    if p == './':
        iFile=p+str(f)
    else:
        iFile = p+"/" + str(f)
    # astype('unicode') does not work as expected

    def coerce_to_unicode(x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)

    #Abrir el fichero .csv y cargarlo en un dataframe de pandas
    ml_dataset = pd.read_csv(iFile)

    #comprobar que los datos se han cargado bien. Cuidado con las cabeceras, la primera linea por defecto la considerara como la que almacena los nombres de los atributos
    # comprobar los parametros por defecto del pd.read_csv en lo referente a las cabeceras si no se quiere lo comentado

    #print(ml_dataset.head(5))

    ml_dataset = ml_dataset[['Especie', 'Ancho de sepalo', 'Largo de sepalo', 'Largo de petalo', 'Ancho de petalo']]


    # Imputa los valores numericos, hay que cambiarlo en funcion de los datos que obtengamos de dataiku


    categorical_features = []
    numerical_features = ['Ancho de sepalo', 'Largo de sepalo', 'Largo de petalo', 'Ancho de petalo']
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


    #se pone un numero por cada una de las columnas que tenemos
    target_map = {'Iris-versicolor': 0, 'Iris-virginica': 1, 'Iris-setosa': 2}
    ml_dataset['__target__'] = ml_dataset['Especie'].map(str).map(target_map)
    del ml_dataset['Especie']

    # Remove rows for which the target is unknown.
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    print(f)
    print(ml_dataset.head(5))

    #el 0.2 significa que da el porcentaje de test 20% y 80% train, el ramdom state indica que va a generar para replicar el mismo train y test asi poder comparar experimentos
    train, test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']])
    print(train.head(5))
    print(train['__target__'].value_counts())
    print(test['__target__'].value_counts())

    drop_rows_when_missing = []
    impute_when_missing = [{'feature': 'Ancho de sepalo', 'impute_with': 'MEAN'},
                       {'feature': 'Largo de sepalo', 'impute_with': 'MEAN'},
                       {'feature': 'Largo de petalo', 'impute_with': 'MEAN'},
                       {'feature': 'Ancho de petalo', 'impute_with': 'MEAN'}]

    # Imputar valores faltantes cuando esten vacias las filas, se imputan con la media
    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]
        print('Dropped missing records in %s' % feature)

    # Explica lo que se hace en este paso
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
        print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))



    rescale_features = {'Ancho de sepalo': 'AVGSTD', 'Largo de sepalo': 'AVGSTD', 'Largo de petalo': 'AVGSTD',
                    'Ancho de petalo': 'AVGSTD'}
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
            print('Feature %s was dropped because it has no variance' % feature_name)
        else:
            print('Rescaled %s' % feature_name)
            train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
            test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale


    trainX = train.drop('__target__', axis=1)
    #trainY = train['__target__']

    testX = test.drop('__target__', axis=1)
    #testY = test['__target__']

    trainY = np.array(train['__target__'])
    testY = np.array(test['__target__'])

    #CUANDO NO ESTAN BALANCEADOS SE PONE EL UNDERSAMPLE.
    #SI HAY POCOS DATOS SE PONE UNDERSAMPLE

    #undersample = elimina aquellas que aparezcan muchas veces para que no me desbalanceen los datos
    #oversample = replicar las instancias que aprecen pocas veces y als copia tantas veces como para que se estabilice | smote crea instancias nuevas haciendo medias entre instancias


    #undersample = RandomUnderSampler(sampling_strategy=0.5)#la mayoria va a estar representada el doble de veces

    #trainXUnder,trainYUnder = undersample.fit_resample(trainX,trainY)
    #testXUnder,testYUnder = undersample.fit_resample(testX, testY)

    # Explica lo que se hace en este paso

    #vamos a realizar un while desde la k minima hasta la k maxima para que genere todas las combinaciones posibles con la k y con la d
    x=1
    #while   x<=int(d):
        #j=int(k)
        #while j<=int(kM):
    clf = KNeighborsClassifier(n_neighbors=5,
                              weights='uniform',
                              algorithm='auto',
                              leaf_size=30,
                              p=2)
    #j=j+2


            # Especifica el peso, en este caso que este balanceado

    clf.class_weight = "balanced"

            # ejecuta el metodo tambien para el test

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
    #editar para que te devuelva el f1_score, precision, recall y accuracy_score con una media, con una media Macro Y Micro
    #Cuando tenemos binario el fscore es None
    #Si es multiclass es macro
    f1 = (f1_score(testY, predictions, average="macro"))
    precision =(precision_score(testY, predictions, average='macro'))
    accuracy = (accuracy_score(testY, predictions))
    recall = (recall_score(testY, predictions, average="macro"))

    #tabla donde te sale todos los parametros
    print(classification_report(testY,predictions))
    print(confusion_matrix(testY, predictions))
    file = open("datos.csv" , 'a')
    file.write( str(j)+", "+ str(x) +", "+ str(precision) +", " + str(recall)+ ", " + str(f1) + "\n")
    print( "-f1-score: \n" + str(f1) +"\n-Precision: \n" + str(precision)+ "\n-Accuracy: \n" + str(accuracy) + "\nRecall: \n" + str(recall))

        #x=x+1
    print("bukatu da")
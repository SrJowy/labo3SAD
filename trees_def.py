# This is a sample Python script.

# Press Mayus+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from datetime import datetime
import getopt
import os
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn import tree
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

mx = 3
mss = 2
msl = 1
p='./'
f="train.csv"
oFile=""
r=0
classifier="TARGET"

def datetime_to_epoch(d):
    return datetime.datetime(d).strftime('%s')
    

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'o:m:s:r:l:p:f:h',['output=','mx=','mss=','msl=','path=','iFile','h'])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-o','--output'):
            oFile = arg
        elif opt == '-m':
            mx = int(arg)
        elif opt ==  '-s':
            mss = int(arg)
        elif opt ==  '-l':
            msl = int(arg)
        elif opt in ('-p', '--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-h','--help'):
            print(' -o outputFile \n -k numberOfItems \n -d distanceParameter \n -p inputFilePath \n -f inputFileName \n ')
            exit(1)
        elif opt in ('-r'):
            r = arg

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

    #ml_dataset = ml_dataset[
    #    ['Largo de sepalo', 'Ancho de sepalo', 'Largo de petalo', 'Ancho de petalo', 'Especie']]
    
    columns = list(ml_dataset.columns)
    ml_dataset = ml_dataset[columns]


    # Se seleccionan los atributos del dataset que se van a utilizar en el modelo


    categorical_features = []
    #numerical_features = ['Largo de sepalo', 'Ancho de sepalo', 'Largo de petalo', 'Ancho de petalo']
    
    numerical_features = list(ml_dataset.columns)
    numerical_features.remove(classifier)
    
    text_features = []
    for feature in categorical_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode) #Actualizar el texto a unicode

    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode) #Actualizar el texto a unicode

    for feature in numerical_features: #M8[ns] --> fecha de 64 bits
        if ml_dataset[feature].dtype == np.dtype('M8[ns]') or ( #Si el tipo del atributo es 'M8[ns]'
                hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')): #o tiene un atributo llamado 'base' y ese atributo es de tipo 'M8[ns]'
            ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature]) #convertimos esa fecha a epoch
        else:
            ml_dataset[feature] = ml_dataset[feature].astype('double') #Cambiamos el tipo el del atributo a double



    #target_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2} #Categorias en las que vamos a encasillar las instancias
    categories = list(ml_dataset[classifier].unique())
    target_map = { str(categories[i]) : i for i in range(0, len(categories))}
    n_cat = len(target_map)
    ml_dataset['__target__'] = ml_dataset[classifier].map(str).map(target_map) #Transformamos el dataset en base a las categorias anteriores, teniendo en cuenta el target o atributo que encasilla las insatancias
    del ml_dataset[classifier] #Borramos el anterior el dataset 

    # Remove rows for which the target is unknown.
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    print(f)
    print(ml_dataset.head(5))


    train, test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']]) #Elegimos la muestra para entrenar el modelo,
    print(train.head(5))                                                                                         #EL 20% sera para test, indice aleatorio de 42
    print(train['__target__'].value_counts())                                                                    #y en base al dataset obtenido antes
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

    #Segun el diccionario anterior, se eliminan los atributos que se hayan dado
    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]
        #print('Dropped missing records in %s' % feature)

    # Segun el diccionario anterior, se imputan los valores mediante la media, la mediana, una categoria, el primer valor o una constante. 
    # Despues se actualizan los valores tanto en el test como en el train 
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
        #print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))


    column_vals = list(ml_dataset.columns)
    column_vals.remove('__target__')
    rescale_features={} #Usar cuando los valores esten desbalanceados
    for i in range(0, len(column_vals)): 
        rescale_features.update({column_vals[i] : 'AVGSTD'})
    
    
    #Se reescalan los valores con respecto al diccionario dado antes por si la muestra se encuentra desbalanceada.
    #Dependiendo del atributo se utiliza MINMAX o la desviacion tipica
    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'MINMAX':
            _min = train[feature_name].min()
            _max = train[feature_name].max()
            scale = _max - _min
            shift = _min
        else:
            shift = train[feature_name].mean()
            scale = train[feature_name].std()
        if scale == 0.: #Si no hay desviacion tipica se ignora ese atributo
            del train[feature_name]
            del test[feature_name]
            #print('Feature %s was dropped because it has no variance' % feature_name)
        else:
            #print('Rescaled %s' % feature_name)
            train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
            test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale


    trainX = train.drop('__target__', axis=1) #Eliminamos la columna con el atributo que clasifica a las instancias
    #trainY = train['__target__']

    testX = test.drop('__target__', axis=1)
    #testY = test['__target__']

    trainY = np.array(train['__target__'])
    testY = np.array(test['__target__'])

    # Explica lo que se hace en este paso
    undersample = RandomUnderSampler(sampling_strategy=0.5)#la mayoria va a estar representada el doble de veces

    trainXUnder,trainYUnder = undersample.fit_resample(trainX,trainY)
    testXUnder,testYUnder = undersample.fit_resample(testX, testY)

    # Calcular el valor del knn
    clf = tree.DecisionTreeClassifier(max_depth=mx,
                                      min_samples_split=mss,
                                      min_samples_leaf=msl)

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

    print(f1_score(testY, predictions, average='macro'))
    print(classification_report(testY,predictions))
    print(confusion_matrix(testY, predictions))
    
    if oFile != "":    
        f = open(oFile, mode='a')
        if (n_cat == 2):
            if os.path.getsize(oFile) == 0:
                f.write("mx, mss, msl, f1_score, recall, precision\n")
            f.write("%s, %s, %s" %(str(mx),str(mss), str(msl)))
            f.write(", %s, %s, %s" %(str(f1_score(testY,predictions, average=None)), str(recall_score(testY,predictions, average=None)), str(precision_score(testY,predictions, average=None)))+ "\n")
        elif (n_cat > 2):
            if os.path.getsize(oFile) == 0:
                   f.write("mx, mss, msl, MACRO_f1_score, MICRO_f1_score, AVG_f1_score, AVG_recall, AVG_precision\n")
            f.write("%s, %s, %s" %(str(mx), str(mss), str(msl)))
            f.write(", %s, %s, %s %s %s" %(str(f1_score(testY,predictions, average='macro')), str(f1_score(testY,predictions, average='micro')), str(f1_score(testY,predictions, average='weighted')), str(recall_score(testY,predictions,average="weighted")), str(precision_score(testY,predictions, average='weighted')))+ "\n")
        f.close()
        
    if r == '1':
        model = "tree.sav"
        saved_model = pickle.dump(clf, open(model,'wb'))
        print('Modelo guardado correctamente')
    
print("bukatu da")
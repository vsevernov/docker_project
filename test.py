import pandas as pd
import requests

data_directory = 'data'


def train(input_data_train, name):
    r = requests.post('http://127.0.0.1:5000/api/ml_models/train', json=input_data_train)
    r1 = requests.get(f'http://127.0.0.1:5000/api/ml_models/model_info/{name}')
    return r1.json()


def predict(input_data_test, name):
    r = requests.put(f'http://127.0.0.1:5000/api/ml_models/predict/{name}', json=input_data_test)
    r = requests.get(f'http://127.0.0.1:5000/api/ml_models/predict/{name}')
    return r.json()

def delete(name):
    r = requests.delete(f'http://127.0.0.1:5000/api/ml_models/predict/{name}')
    return r.json()


print('Привет!', 'На нашем сервисе доступны следующие архитектуру ML моделей:', sep='\n')
r_model_class = requests.get('http://127.0.0.1:5000/api/ml_models/model_class')
print(r_model_class.json())
task = str(input('Что надо сделать (train, predict, delete, update, show exist models info)'))

if task == 'train':
    model_name = str(input('Какую модель будешь использовать (RandomForestClassifier, lightgbm, DecisionTreeRegressor'))
    if model_name == 'lightgbm':
        df_train = pd.read_csv('data/titanic_train.csv')
        data_train = df_train.to_json()
        df_test = pd.read_csv('data/titanic_test.csv')
        data_path = 'titanic_train.csv'
        params = {
            'learning_rate': 2,
            'max_depth': 10,
            'random_state': 42
        }
        features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        target = ['Survived']
        model_name = str(input('Введите название модели'))
        input_data_train = {
            'params': params, 'data': data_train,
            'features': features, 'target': target,
            'model_name': 'lightgbm', 'dataset': data_path,
            'name': model_name}

        print(train(input_data_train, model_name))

    if model_name == 'RandomForestClassifier':
        df_train = pd.read_csv('data/titanic_train.csv')
        data_train = df_train.to_json()
        df_test = pd.read_csv('data/titanic_test.csv')
        data_path = 'titanic_train.csv'
        params = {
             'max_depth': 2,
             'random_state': 10}

        features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        target = ['Survived']
        model_name = str(input('Введите название модели'))
        input_data_train = {
            'params': params, 'data': data_train,
            'features': features, 'target': target,
            'model_name': 'RandomForestClassifier', 'dataset': data_path,
            'name': model_name}

        print(train(input_data_train, model_name))

    if model_name == 'DecisionTreeRegressor':
        df_train = pd.read_csv('data/boston_train.csv')
        data_train = df_train.to_json()
        data_path = 'boston_train.csv'
        params = {
            'max_depth': 2
        }

        features = ['lstat', 'rm']
        target = ['medv']
        model_name = str(input('Введите название модели'))
        input_data_train = {
            'params': params, 'data': data_train,
            'features': features, 'target': target,
            'model_name': 'DecisionTreeRegressor', 'dataset': data_path,
            'name': model_name}

        print(train(input_data_train, model_name))


elif task == 'predict':
    model_name = str(input('Какую модель будешь использовать (RandomForestClassifier, lightgbm, DecisionTreeRegressor'))
    if model_name == 'lightgbm':
        print("Предсказывать будем на датасете titanic_test.csv")
        df_test = pd.read_csv('data/titanic_test.csv')
        features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        data_test = df_test[features]
        data_test = data_test.to_json()
        input_data_test = {'data': data_test}
        print("Характеристики доступных моделей")
        r = requests.get('http://127.0.0.1:5000/api/ml_models/models_info')
        print(r.json())
        name = input("Введите имя модели, предсказания которой хотите получить")
        print(predict(input_data_test, name))

    if model_name == 'RandomForestClassifier':
        print("Предсказывать будем на датасете titanic_test.csv")
        df_test = pd.read_csv('data/titanic_test.csv')
        features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        data_test = df_test[features]
        data_test = data_test.to_json()
        input_data_test = {'data': data_test}
        print("Характеристики доступных моделей")
        r = requests.get('http://127.0.0.1:5000/api/ml_models/models_info')
        print(r.json())
        name = input("Введите имя модели, предсказания которой хотите получить")
        print(predict(input_data_test, name))

    if model_name == 'DecisionTreeRegressor':
        print("Предсказывать будем на датасете titanic_test.csv")
        df_test = pd.read_csv('data/boston_test.csv')
        features = ['lstat', 'rm']
        data_test = df_test[features]
        data_test = data_test.to_json()
        input_data_test = {'data': data_test}
        print("Характеристики доступных моделей")
        r = requests.get('http://127.0.0.1:5000/api/ml_models/models_info')
        print(r.json())
        name = input("Введите имя модели, предсказания которой хотите получить")
        print(predict(input_data_test, name))

elif task == 'delete':
    print("Характеристики доступных моделей")
    r = requests.get('http://127.0.0.1:5000/api/ml_models/models_info')
    name = str(input("Введите имя модели которую хотите удалить"))
    print(delete(name))

elif task == 'show exist models info':
    print("Характеристики доступных моделей")
    r = requests.get('http://127.0.0.1:5000/api/ml_models/models_info')
    print(r.json())

elif task == 'update':
    old_name = str(input("Старое имя модели"))
    new_name = str(input('Новое имя модели'))
    input_data = {'old_name': old_name, 'new_name': new_name}
    r = requests.post('http://127.0.0.1:5000/api/ml_models/update', json=input_data)
    print(r.json())
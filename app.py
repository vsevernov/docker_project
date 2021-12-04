from flask import Flask, jsonify
from flask_restx import Api, Resource
from ml_model import model

app = Flask(__name__)
api = Api(app)


@api.route('/api/ml_models/train')
class RestApi(Resource):
    def get(self):
        """
        Возвращает характеристики текущей модели на стадии обучения и отчет с
        характеристиками всех доступных обученных ранее моделей.
        Структура отчета:
        (
        dataset: на каком датасете обучалась модель
        features: фичи на которых обучаласть модель (порядок с которыми подавались в модель - сохранен)
        target: целевая переменная
        metric: метрика качества
        model_name: тип модели (RandomForestClassifier, lightgbm,...)
        name: имя модели как ее назвал пользователь
        exist_models: список всех обученных моделей
        exist_models_type: доступные классы моделей
        models_description: характеристики всех обученных моделей

        :return:
        """
        return jsonify({"info": model.available_models_info()})

    def post(self):
        """
        Функция передает данные для обучения в фунцию model.model_train (формат json)
        Данные включают в себя:
        params: параметры модели
        data: тренировочный датасет (формат json)
        features: фичи, которые передаются в модель (важет порядок)
        target: целевая переменная
        model_name: тип модели (RandomForestClassifier, lightgbm,...)
        dataset: название датасета на котором обучалась модель
        name: имя модели (с таким именем модель и будет сохранятся)
        :return: экземпляр класса
        """
        return model.model_train(api.payload)


@api.route('/api/ml_models/model_class')
class RestApi(Resource):
    def get(self):
        """
        Функция для получения доступных классов моделей
        :return: доступные классы моделей
        """
        return jsonify({"info": model.available_сlass_model()})


@api.route('/api/ml_models/model_info/<string:name>')
class RestApi(Resource):
    def get(self, name):
        """
        Функция для получения информации о заданной модели
        :return: информация о модели
        """
        return jsonify({"info": model.pretrained_model_info(name)})


@api.route('/api/ml_models/models_info')
class RestApi(Resource):
    def get(self):
        """
        Функция для получения информации о доступных обученных
        ранее моделей
        :return: доступные классы моделей
        """
        return jsonify({"info": model.available_models_info()})



@api.route('/api/ml_models/update')
class MLModelsName(Resource):
    def post(self):
        """
        Функция используется для передачи данных в модель на стадии предсказания
        Данные включают в себя:
        data: обучающий датасет (формат json)
        :param name: название модели, которую будем исспользовать для предсказания
        :return: предскзания модели в формате dict
        """
        return jsonify({"info": model.update_model_name(api.payload)})

    def get(self):
        return jsonify({"info": model.update_model_info()})



@api.route('/api/ml_models/predict/<string:name>')
class MLModelsName(Resource):
    def put(self, name):
        """
        Функция используется для передачи данных в модель на стадии предсказания
        Данные включают в себя:
        data: обучающий датасет (формат json)
        :param name: название модели, которую будем исспользовать для предсказания
        :return: предскзания модели в формате dict
        """
        return model.predict_model(api.payload, name)

    def get(self, name):
        """
        Функция используется для получения предсказаний модели с названием name
        :param name: название модели, которую будем исспользовать для предсказания
        :return: предсказания модели в формате json
        """
        # return jsonify({"prediction": list(map(int, model.get_model_predict()))})
        return jsonify({"prediction": model.get_model_predict()})

    def delete(self, name):
        """
        Функция удаляет обученную модель из доступных моделей (из ./model)
        :param name: название модели, которую необходимо удалить
        :return: message
        """
        message = model.delete_model(name)
        return message


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = "5000")

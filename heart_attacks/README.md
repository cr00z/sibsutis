# Задание

На сайте kaggle по ссылке https://www.kaggle.com/imnikhilanand/heart-attackprediction/home располагаются реальные данные по сердечной заболеваемости, собранные различными медицинскими учреждениями.

Каждый человек представлен 14ю характеристиками и полем goal, которое показывает наличие болезни сердца, поле принимает значение от 0 до 4 (0 – нет болезни).

Требуется имеющиеся данные разбить на обучающую и тестовую выборки в процентном соотношении 70 к 30. После чего по обучающей выборке необходимо построить решающее дерево. Для построения дерева можно пользоваться любыми существующими средствами.

Кроме того, для построения дерева необходимо будет решить задачу выделения информативных решающих правил относительно имеющихся числовых признаков. Разрешается использовать уже реализованные решающие деревья из известных библиотек (например, scikit-learn для Python), либо реализовывать алгоритм построения дерева самостоятельно (все необходимые алгоритмы представлены в теории по ссылке).

В качестве результата работы необходимо сделать 10 случайных разбиений исходных данных на обучающую и тестовую выборки, для каждой построить дерево и протестировать, после чего построить таблицу, в которой указать процент правильно классифицированных данных. Полученную таблицу необходимо включить в отчёт по лабораторной работе.

# Описание датасета:

1. 3 (**age**): age in years
2. 4 (**sex**): sex (1 = male; 0 = female)
3. 9 (**cp**): chest pain type
    - Value 1: typical angina
    - Value 2: atypical angina
    - Value 3: non-anginal pain
    - Value 4: asymptomatic
4. 10 (**trestbps**): resting blood pressure (in mm Hg on admission to the hospital)
5. 12 (**chol**): serum cholestoral in mg/dl
6. 16 (**fbs**): (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
7. 19 (**restecg**): resting electrocardiographic results
    - Value 0: normal
    - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
8. 32 (**thalach**): maximum heart rate achieved
9. 38 (**exang**): exercise induced angina (1 = yes; 0 = no)
10. 40 (**oldpeak**): ST depression induced by exercise relative to rest
11. 41 (**slope**): the slope of the peak exercise ST segment
    - Value 1: upsloping
    - Value 2: flat
    - Value 3: downsloping
12. 44 (**ca**): number of major vessels (0-3) colored by flourosopy
13. 51 (**thal**): 3 = normal; 6 = fixed defect; 7 = reversable defect
14. 58 (**num**) (the predicted attribute): diagnosis of heart disease (angiographic disease status)
    - Value 0: < 50% diameter narrowing
    - Value 1: > 50% diameter narrowing (in any major vessel: attributes 59 through 68 are vessels)
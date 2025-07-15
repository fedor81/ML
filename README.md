# 🤖 ML

Репозиторий с решенными задачами по машинному обучению

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" alt="Python">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Numpy-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Seaborn-5B8CBF?logo=seaborn&logoColor=white" alt="Seaborn">
  <img src="https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white" alt="Jupyter">
</p>

## 📂 Содержание

- [Лучшие результаты](#-лучшие-результаты)
- [Проекты 2-го курса](#-проекты-2-го-курса)
  - ⚽ [FootballPrediction: прогнозирование результатов футбольных матчей](#-footballprediction-прогнозирование-результатов-футбольных-матчей)
  - 🎵 [Предсказание популярности песен (Regression)](#-предсказание-популярности-песен-regression)
  - 🚢 [Titanic: предсказание виживаемости](#-titanic-предсказание-виживаемости)
- [Проекты 1-го курса](#-проекты-1-го-курса)
  - 🐘 [Классификация изображений (Human/Animal)](#-классификация-изображений-humananimal)

## 🏆 Лучшие результаты

| Проект               | Метрика          | Значение | Модель        |
| -------------------- | ---------------- | -------- | ------------- |
| ⚽ FootballPrediction | Confusion Matrix |          | Нейросеть     |
| 🎵 Songs Popularity   | R²               | 0.763    | Random Forest |
| 🚢 Titanic            | Accuracy         | 0.77     | Нейросеть     |
| 🐘 Human/Animal       | Accuracy         | 81.33%   | CNN           |

## 🎓 Проекты 2-го курса

### ⚽ FootballPrediction: прогнозирование результатов футбольных матчей

[![Folder](https://img.shields.io/badge/Folder-%20basics--deep--learning--and--AI/football/-informational?style=flat&logo=openmoji&color=blue)](./basics-deep-learning-and-AI/football/)

Основные возможности:

- **Предсказание счета** матча на основе исторических данных
- Анализ влияния различных факторов на исход игры
- **Графический интерфейс** для удобного взаимодействия

<br/>

| Сравнение              | Home Wins | Draw  | Away Wins |
| ---------------------- | --------- | ----- | --------- |
| Истинные значения      | 243       | 121   | 136       |
| Предсказанные значения | 274       | 153   | 73        |
| Precision              | 0.631     | 0.314 | 0.534     |
| Recall                 | 0.712     | 0.397 | 0.288     |
| F1                     | 0.669     | 0.350 | 0.373     |

### 🎵 Предсказание популярности песен (Regression)

[![Open in Kaggle](https://img.shields.io/badge/Kaggle-Open-blue?logo=kaggle)](https://www.kaggle.com/code/laroxyss/songs-korolev-fedor)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](./Songs%20Korolev%20Fedor.ipynb)

**Задача**: Прогнозирование популярности песен на основе их характеристик

```python
# Основные этапы:

1. EDA и визуализация
2. Feature engineering
3. Эксперименты с моделями
```

**Лучшие результаты**:

- ✅ MAE: 7.47
- ✅ RMSE: 10.84
- ✅ R²: 0.763

### 🚢 Titanic: предсказание виживаемости

[![Open in Kaggle](https://img.shields.io/badge/Kaggle-Open-blue?logo=kaggle)](https://www.kaggle.com/code/laroxyss/titanic-korolev-fedor-at07)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](./Titanic%20Korolev%20Fedor%20AT07.ipynb)

**Задача**: Предсказание выживания пассажиров Титаника

**Особенности**:

- 🔍 Анализ признаков
- 🧩 Кросс-валидация
- 🤖 Эксперименты с архитектурой нейросети

**Топ-модель 13**:

- 🎯 Accuracy: `0.77`
- 🎯 Precision: `0.74`
- 🎯 Recall: `0.69`

## 📚 Проекты 1-го курса

### 🐘 Классификация изображений (Human/Animal)

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](./human_animal.ipynb)

**Задача**: Определить, человек или животное на изображении  

**Архитектура CNN**:

```
Conv2D → MaxPooling → Dropout → BatchNorm → Linear
```

**Результаты**:

| Эпоха | Accuracy | Правильно |
| ----- | -------- | --------- |
| 4     | 81.33%   | 122/150   |
| 6     | 79.33%   | 119/150   |

## 📜 License  

This project is licensed under the **MIT License** - see the [LICENSE](./LICENSE) file for details.
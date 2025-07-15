import os
import sys
import torch
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QFormLayout,
    QMessageBox,
    QDateEdit,
)
from PySide6.QtCore import QDate

from . import mappings
from ..files import models, dataset_load


class FootballPredictorGUI(QMainWindow):
    def __init__(self, model, team_map, tournament_map, city_map, country_map, scorer_map):
        super().__init__()
        self.model = model
        self.mappings = {
            "teams": team_map,
            "tournaments": tournament_map,
            "cities": city_map,
            "countries": country_map,
            "scorers": scorer_map,
        }

        self.init_ui()
        self.setWindowTitle("Football Match Predictor")

    def init_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout()

        # Форма ввода данных
        form = QFormLayout()

        # Выпадающие списки для категориальных признаков
        self.home_team_combo = QComboBox()
        self.home_team_combo.addItems(sorted(self.mappings["teams"].keys()))

        self.away_team_combo = QComboBox()
        self.away_team_combo.addItems(sorted(self.mappings["teams"].keys()))

        self.tournament_combo = QComboBox()
        self.tournament_combo.addItems(sorted(self.mappings["tournaments"].keys()))

        self.city_combo = QComboBox()
        self.city_combo.addItems(sorted(self.mappings["cities"].keys()))

        self.country_combo = QComboBox()
        self.country_combo.addItems(sorted(self.mappings["countries"].keys()))

        self.date_edit = QDateEdit()

        self.neutral_edit = QComboBox()
        self.neutral_edit.addItems(["False", "True"])

        # Добавление полей в форму
        form.addRow("Home Team:", self.home_team_combo)
        form.addRow("Away Team:", self.away_team_combo)
        form.addRow("Tournament:", self.tournament_combo)
        form.addRow("Country:", self.country_combo)
        form.addRow("City:", self.city_combo)
        form.addRow("Neutral:", self.neutral_edit)
        form.addRow("Date:", self.date_edit)

        # Кнопка предсказания
        predict_btn = QPushButton("Predict Match Outcome")
        predict_btn.clicked.connect(self.predict)

        # Поле для вывода результатов
        self.result_label = QLabel("Prediction will appear here")
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        # Сборка интерфейса
        layout.addLayout(form)
        layout.addWidget(predict_btn)
        layout.addWidget(self.result_label)
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def predict(self):
        history_count = 5
        goals_count = 10
        num_features = 11
        history_size = (1, history_count, num_features)
        goals_size = (1, goals_count)

        # try:
        if True:
            tensor_data = {
                "match_features": torch.tensor([self.get_match_features()], dtype=torch.long),
                "home_history": torch.zeros(history_size, dtype=torch.float32),
                "away_history": torch.zeros(history_size, dtype=torch.float32),
                "goals_times": torch.zeros(goals_size, dtype=torch.float32),
                "goals_scorers": torch.zeros(goals_size, dtype=torch.long),
            }

            # Предсказание
            with torch.no_grad():
                prediction = self.model(tensor_data)
                home_pred = round(prediction[0][0].item(), 1)
                away_pred = round(prediction[0][1].item(), 1)

            # Определение исхода
            if home_pred > away_pred:
                outcome = "Home Win"
            elif home_pred < away_pred:
                outcome = "Away Win"
            else:
                outcome = "Draw"

            # Вывод результатов
            result_text = (
                f"Predicted Score: {home_pred:.1f} - {away_pred:.1f}\n"
                f"Outcome: {outcome}\n"
                f"Total Goals: {home_pred + away_pred:.1f}"
            )
            self.result_label.setText(result_text)

        # except Exception as e:
        #     QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")

    def get_match_features(self):
        """Получает закодированные признаки матча"""
        date = self.date_edit.date()

        return [
            self.mappings["teams"][self.home_team_combo.currentText()],
            self.mappings["teams"][self.away_team_combo.currentText()],
            self.mappings["tournaments"][self.tournament_combo.currentText()],
            self.mappings["cities"][self.city_combo.currentText()],
            self.mappings["countries"][self.country_combo.currentText()],
            date.year(),
            date.month(),
            date.day(),
            1 if self.neutral_edit.currentText() == "True" else 0,
        ]


def run_gui(model, data_mappings):
    app = QApplication(sys.argv)
    gui = FootballPredictorGUI(
        model,
        data_mappings["team_map"],
        data_mappings["tournament_map"],
        data_mappings["city_map"],
        data_mappings["country_map"],
        data_mappings["scorer_map"],
    )
    gui.show()
    sys.exit(app.exec_())


ROOT_DIR = "./basics-deep-learning-and-AI/football/ui"


def create_model():
    """Создает модель и загружает веса"""
    print("Loading model...")
    info = dataset_load.get_dataset_info()
    model = models.AdvancedFootballModel(
        info["num_teams"],
        info["num_tournaments"],
        info["num_cities"],
        info["num_countries"],
        info["num_scorers"],
    )

    state_dict = torch.load(os.path.join(ROOT_DIR, "weights.pt"))
    model.load_state_dict(state_dict)

    return model


if __name__ == "__main__":
    test_mappings = {
        "team_map": {"Team A": 0, "Team B": 1},
        "tournament_map": {"League": 0, "Cup": 1},
        "city_map": {"London": 0, "Paris": 1},
        "country_map": {"England": 0, "France": 1},
        "scorer_map": {"Player1": 0, "Player2": 1},
    }

    print("Loading mappings...")
    data_mappings = mappings.create_mappings()
    test_mappings.update(data_mappings)

    model = create_model()
    run_gui(model, test_mappings)

import torch
import torch.nn as nn
import torch.nn.functional as F


class FootballMatchPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size + 16, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, data):
        return self.fc(data)


class AdvancedFootballModel(nn.Module):
    def __init__(
        self,
        num_teams,
        num_tournaments,
        num_cities,
        num_countries,
        num_scorers,
        history_length=5,
        goals_length=10,
    ):
        super().__init__()

        # Размерности
        self.team_emb_size = 16
        self.tournament_emb_size = 8
        self.city_emb_size = 8
        self.country_emb_size = 8
        self.scorer_emb_size = 8

        self.features_count = 9
        self.history_input_size = self.features_count + 2  # +2 для home_score, away_score
        self.history_hidden_size = 32

        # Эмбеддинги для категориальных признаков
        self.team_embed = nn.Sequential(
            # Добавляем 1 так как кодировали LabelEncoderStartingFromOne
            nn.Embedding(num_teams + 1, self.team_emb_size),
            nn.LayerNorm(self.team_emb_size),
        )

        self.tournament_embed = nn.Sequential(
            nn.Embedding(num_tournaments, self.tournament_emb_size),
            nn.LayerNorm(self.tournament_emb_size),
        )

        self.city_embed = nn.Sequential(
            nn.Embedding(num_cities, self.city_emb_size),
            nn.LayerNorm(self.city_emb_size),
        )

        self.country_embed = nn.Sequential(
            nn.Embedding(num_countries, self.country_emb_size),
            nn.LayerNorm(self.country_emb_size),
        )

        # Обработка истории (5 матчей × 11 фичей)
        self.history_encoder = nn.LSTM(
            input_size=self.history_input_size,
            hidden_size=self.history_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        # Обработка голов
        goals_out_size = 32
        self.goal_processor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * 5, goals_out_size),  # 5 = 10//2 после maxpool
        )

        # # Обработка игроков
        scorer_out_size = 16
        # Добавляем 1 так как кодировали LabelEncoderStartingFromOne
        self.scorer_embed = nn.Embedding(num_scorers + 1, scorer_out_size)
        self.scorer_attention = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)

        # Объединяющий блок
        self.final = nn.Sequential(
            nn.Linear(
                self.team_emb_size * 2
                + self.tournament_emb_size
                + self.city_emb_size
                + self.country_emb_size
                + self.history_hidden_size * 2
                + goals_out_size
                + scorer_out_size,
                256,
            ),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 2),  # [home_score, away_score]
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
                nn.init.constant_(module.bias, 0.1)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.fill_(0.1)

    def forward(self, x):
        home_emb = self.team_embed(x["match_features"][:, 0])
        away_emb = self.team_embed(x["match_features"][:, 1])
        tournament_emb = self.tournament_embed(x["match_features"][:, 2])
        city_emb = self.city_embed(x["match_features"][:, 3])
        country_emb = self.country_embed(x["match_features"][:, 4])

        # История команд
        home_history, _ = self.history_encoder(x["home_history"])
        away_history, _ = self.history_encoder(x["away_history"])

        # Агрегация истории (берем последнее состояние)
        home_history = home_history[:, -1, :]
        away_history = away_history[:, -1, :]

        # Обработка голов
        time_feat = self.goal_processor(x["goals_times"].unsqueeze(1))

        # Внимание к бомбардирам
        scorers_emb = self.scorer_embed(x["goals_scorers"])
        scorer_feat, _ = self.scorer_attention(scorers_emb, scorers_emb, scorers_emb)
        scorer_feat = scorer_feat.mean(dim=1)

        # Объединение всех признаков
        combined = torch.cat(
            [
                home_emb,
                away_emb,
                tournament_emb,
                city_emb,
                country_emb,
                home_history,
                away_history,
                time_feat,
                scorer_feat,
            ],
            dim=1,
        )

        return self.final(combined)

import numpy as np
import pandas as pd
from src.utils.state_encoder import StateEncoder


class CollegeEnv:

    def __init__(self, csv_file, max_questions=10):
        self.done = None
        self.steps = None
        self.asked_questions = None
        self.known_answers = None
        self.current_row = None
        self.df = pd.read_csv(csv_file)
        self.encoder = StateEncoder(df=self.df)
        self.max_questions = max_questions
        self.feature_columns = [col for col in self.df.columns if col not in ["Predicted_College", "Predicted_Branch"]]
        self.reset()

    def reset(self):
        self.current_row = self.df.sample(1).iloc[0]
        self.known_answers = {col: None for col in self.feature_columns}
        self.asked_questions = set()
        self.steps = 0
        self.done = False
        return self._build_state()

    def _build_state(self):
        state_parts = []
        mask = []
        for col in self.feature_columns:
            if self.known_answers[col] is None:
                state_parts.append(np.zeros(self._feature_size(col)))
                mask.append(0)
            else:
                state_parts.append(self._encode_feature(col, self.known_answers[col]))
                mask.append(1)
        state_vector = np.concatenate(state_parts + [np.array(mask)])
        return state_vector

    def _encode_feature(self, col, answer):
        if col == "Rank":
            return self.encoder.encode_rank(answer)
        elif col == "Category":
            return self.encoder.encode_category(answer)
        elif col == "Gender":
            return self.encoder.encode_gender(answer)
        elif col == "State":
            return self.encoder.encode_state(answer)
        elif col in ["Placement",	"Research",	"Startup"]:
            return  self.encoder.encode_interest_c2(answer)
        else:
            return self.encoder.encode_interest(answer)

    def _feature_size(self, col):
        if col == "Rank":
            return 1
        elif col == "Category":
            return len(self.encoder.categories)
        elif col == "Gender":
            return len(self.encoder.genders)
        elif col == "State":
            return len(self.encoder.states)
        else:
            return 1

    def step(self, action):
        if self.done:
            raise Exception("Episode Finished. Call reset()")
        reward = 0

        if action <len(self.feature_columns):
            col = self.feature_columns[action]

            if col in self.asked_questions:
                reward = -2
            else:
                self.known_answers[col] = self.current_row[col]
                self.asked_questions.add(col)
                reward = -0.5

            self.steps += 1

            if self.steps > self.max_questions:
                self.done = True

            else:
                predicted_college =self._rule_based_predict()
                true_college = self.current_row['Predicted_College']
                if predicted_college == true_college:
                    reward = 30
                else:
                    reward = -20

                self.done = True

        return self._build_state(),reward, self.done

    def _rule_based_predict(self):
        filtered_df = self.df.copy()

        for col, value in self.known_answers.items():
            if value is not None:
                filtered_df = filtered_df[filtered_df[col] == value]

        if len(filtered_df) == 0:
            return self.df["Predicted_College"].mode()[0]

        return filtered_df["Predicted_College"].mode()[0]

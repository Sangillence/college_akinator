import pandas as pd
import numpy as np


class StateEncoder:
    def __init__(self,df):
        self.df = df
        self.categories = df['Category'].unique().tolist()
        self.genders = df['Gender'].unique().tolist()
        self.states = df['State'].unique().tolist()

        self.interestMap = {
            "Not_Interested": 0,
            "Somewhat_Interested": 1,
            "Interested": 2,
            "Much_Interested": 3,
            "Very_Interested": 4,
            "High":2,
            "Medium":0,
            "Low":1
        }

        self.max_rank = df['Rank'].max()

    def encode_rank(self, rank):
            return  np.array([rank/self.max_rank])

    def encode_category(self, category):
        vec = np.zeros(len(self.categories))
        idx = self.categories.index(category)
        vec[idx] = 1
        return vec

    def encode_gender(self, gender):
        vec = np.zeros(len(self.genders))
        idx = self.genders.index(gender)
        vec[idx] = 1
        return vec

    def encode_state(self, state):
        vec = np.zeros(len(self.states))
        idx = self.states.index(state)
        vec[idx] = 1
        return vec

    def encode_interest(self, interest):
        return  np.array([self.interestMap[interest]/4.0])
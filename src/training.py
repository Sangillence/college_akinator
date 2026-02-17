import numpy as np
from src.env.college_env import CollegeEnv
from src.agents.dqn_agent import  DQNAgent
from src.model.tfq_network import TFQNetwork
from src.model.q_network import QNetwork
import matplotlib.pyplot as plt
import torch
import tensorflow as tf

def train():
    env = CollegeEnv("S:\\My Projects\\akinator\\data\\raw\\states.csv")
    state_size = len(env.reset())
    action_size = len(env.feature_columns) +1
    agent = DQNAgent(state_size, action_size)
    num_episodes = 1000
    rewards_history = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(np.array(state))
            next_state, reward, done, = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward
        rewards_history.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode: {episode}, Reward: {total_reward}")

    print("Training Complete!")
    torch.save(agent.q_network.state_dict(), "output/dqn_college_model.pth")
    print("Model saved.")
    evaluate(agent,env)
    convert_to_tflite(state_size, action_size)
    return rewards_history


def evaluate(agent, env, episodes=50):
    agent.epsilon = 0.0  # pure exploitation
    total_rewards = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            state, reward, done = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"Average Evaluation Reward: {avg_reward:.2f}")

def convert_to_tflite(state_size, action_size):
    print("Converting PyTorch model to TFLite...")
    pytorch_model = QNetwork(state_size, action_size)
    pytorch_model.load_state_dict(torch.load("output/dqn_college_model.pth", map_location="cpu"))
    pytorch_model.eval()
    tf_model = TFQNetwork(state_size, action_size, pytorch_model)
    tf.saved_model.save(tf_model, "output/saved_model", signatures={"serving_default": tf_model.forward})
    print("TensorFlow SavedModel exported to 'saved_model'")
    converter = tf.lite.TFLiteConverter.from_saved_model("output/saved_model")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # optional quantization
    tflite_model = converter.convert()

    with open("output/dqn_college_model.tflite", "wb") as f:
        f.write(tflite_model)

    print("TFLite model saved as 'dqn_college_model.tflite'")


if __name__ == "__main__":
   rewards_history =  train()
   plt.plot(rewards_history)
   plt.xlabel("Episode")
   plt.ylabel("Total Reward")
   plt.title("Training Reward Curve")
   plt.show()

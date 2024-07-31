
import minigrid
import torch
import gymnasium as gym
import torch.nn as nn
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, RGBImgObsWrapper, FlatObsWrapper
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava16ChatHandler
import base64
from PIL import Image
import warnings
import numpy as np
import io


warnings.filterwarnings("ignore")


#########################################################################################

llm = Llama(
      model_path="ggml-model-Q5_K_M.gguf",
      n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)

output = llm(
      "Q: Name the planets in the solar system? A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion

print(output)

#########################################################################################


#########################################################################################

# Setup Gym environment
env = gym.make("MiniGrid-Empty-5x5-v0")
env = RGBImgPartialObsWrapper(env)
env = ImgObsWrapper(env)
env = Monitor(env, "./a2c_cartpole_tensorboard/")
state, _ = env.reset()

# Initialize Llama model
chat_handler = Llava16ChatHandler(clip_model_path="mmproj-model-f16.gguf")
llm = Llama(
  model_path="llava-v1.6-mistral-7b.Q3_K_XS.gguf",
  chat_handler=chat_handler,
  n_ctx=4096, # n_ctx should be increased to accommodate the image embedding
)

# Function to convert NumPy array to base64 data URI
def ndarray_to_base64_data_uri(ndarray):
    img = Image.fromarray(ndarray)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{base64_data}"

# Convert the NumPy array image (from Gym environment) to a data URI
data_uri = ndarray_to_base64_data_uri(state)

# Messages with image data URI
messages = [
    {"role": "system", "content": "You are GPT-4 who perfectly describes images and provides feedback for reinforcement learning agent."},
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": "The image is the partial view of the reinforcement learning agent of the 5x5 gridworld. Position of the goal is down to the right. The available actions are 1. Go Left, 2. Go Right, 3. Go Up, and 4. Go Down. Please select an appropriate action for the reinforcement learning agent."}
        ]
    }
]

# Get response from the model
#response = llm.create_chat_completion(messages=messages)

try:
    # Your main code logic
    response = llm.create_chat_completion(messages=messages)
    print(response)
finally:
    # Explicitly close the Llama object or any related resources
    llm.close()

print(response)

#########################################################################################
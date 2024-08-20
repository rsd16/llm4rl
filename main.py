# %%
import PIL
import base64
import requests
import openai
import minigrid
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt


from io import BytesIO
from minigrid.wrappers import RGBImgPartialObsWrapper, ViewSizeWrapper

# %%
ACTIONS = [
    "Turn Clockwise",
    "Turn Anti Clockwise",
    "Move forward",
    "Pick up an object",
    "Drop the Object",
    "Toggle Something",
]
ACTIONS = [f"{x}" for i, x in enumerate(ACTIONS)]
ACTIONS = "\n".join(ACTIONS)
DIRECTION = ("Right", "Down", "Left", "Up")

# %%
env = gym.make("MiniGrid-ObstructedMaze-1Dlhb-v0", render_mode="rgb_array")
env = ViewSizeWrapper(env, agent_view_size=3)
env = RGBImgPartialObsWrapper(env)

# %%
state, info = env.reset()
render = env.render()
fig, axs = plt.subplots(2)
axs[0].imshow(render)
axs[1].imshow(state["image"])
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(f"Direction: {DIRECTION[state['direction']]} | {state['mission']}")
fig.tight_layout()

# %%
def b64(image: np.ndarray) -> str:
    buffered = BytesIO()
    PIL.Image.fromarray(image).save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# %%
SYSTEM = """
You Are GPT-4 State of the art language and vision model
You are Reinforcement Learning Agent, Living in Maze. 
Your top priority is to solve this maze based on current observation and plan for actions!
""".strip()

# %%
PROMPT = f"""
Environment Description and Objective:
- Gray Cells are walls.
- Black Cells are empty.
- The Red Triangle is me.
- Mission is to {state['mission']} 
- the ball is in another room and need to find a key to unlock the door.
- there is a mystery box!

Available Actions:
{ACTIONS}

in order to solve this maze and find the blue ball and my current viewpoint [img-10] i would take following actions:
1.
""".strip()

# %%
datas = {
    "prompt": PROMPT,
    "n_predict": -1,
    "stream": False,
    "image_data": [
        {
            "id": 10,
            "data": b64(state["image"]),
        }
    ],
}
data = {
    "prompt": PROMPT,
    "n_predict": -1,
    "stream": False,
}

# %%
response = requests.post(
    "http://127.0.0.1:8080/completion",
    headers={"Content-Type": "application/json"},
    json=data,
)
resp = response.json()

# %%
print(resp["content"])

# %%


# %%


# llama-server -m E:\llava1.6\llava-v1.6-mistral-7b.Q3_K_XS.gguf -p 8080

# llama-server -m E:\minicpm\ggml-model-Q5_K_M.gguf -p 8080
E:\minicpm
minicpmv-cli -m E:\minicpm\ggml-model-Q5_K_M.gguf --mmproj E:\minicpm\mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image E:\minicpm\Mental-Strong-Women-min.jpg -p "What is in the image?"
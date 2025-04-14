import socket
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from shard_utils import split_model

# Load model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
(wte, wpe, drop, part1_blocks), _, _ = split_model(model, cut_layer_idx=3)

# Prepare blocks
for block in part1_blocks:
    block.eval().to(dtype=torch.float32)

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

examples = [
    "The cat is sitting",
    "Tomorrow we will go",
    "The weather is very",
    "She is reading a",
    "This place looks so",
    "The project will be",
    "They are working on",
    "He went to the market",
    "I love eating sushi",
    "Our team is ready",
    "The book was about",
    "He started running when",
    "A dog barked loudly",
    "They built a new",
    "She wants to learn",
    "We should arrive by",
    "The car stopped suddenly",
    "It will rain tomorrow",
    "He was thinking about",
    "The light turned red"
]

# Loop through each input
HOST = '192.168.1.45'  # Ubuntu IP
PORT = 5002

for i, input_text in enumerate(examples):
    print(f"\n[{i+1}/10] input_text: {input_text}")
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(dtype=torch.long)

    with torch.no_grad():
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        inputs_embeds = wte(input_ids) + wpe(position_ids)
        hidden_state = drop(inputs_embeds)

        for block in part1_blocks:
            hidden_state = block(hidden_state)[0]

    data = hidden_state.numpy().astype("float32").tobytes()

    # Send to Ubuntu
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(data)
        print(f"Sent tensor shape {hidden_state.shape}")
    except Exception as e:
        print(f"Error sending: {e}")

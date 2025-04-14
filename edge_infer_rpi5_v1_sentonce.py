import socket
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from shard_utils import split_model

# Load and split model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
(wte, wpe, drop, part1_blocks), _, _ = split_model(model, cut_layer_idx=3)  # blocks is list

# Convert each block to eval mode
for block in part1_blocks:
    block.eval().to(dtype=torch.float32)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
input_text = "Today is a nice"
print("input_text:",input_text)
input_ids = tokenizer.encode(input_text, return_tensors="pt")
input_ids = input_ids.to(dtype=torch.long)

# Embed + Run blocks manually
with torch.no_grad():
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    inputs_embeds = wte(input_ids) + wpe(position_ids)
    hidden_state = drop(inputs_embeds)

    # ❗ Run each layer manually and keep only the first output
    for block in part1_blocks:
        hidden_state = block(hidden_state)[0]

# Send to Ubuntu
HOST = '192.168.1.45'  # Your Ubuntu IP
PORT = 5002
data = hidden_state.numpy().astype("float32").tobytes()

print(f"📤 Sending tensor: {hidden_state.shape}")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(data)
print("✅ Sent successfully")

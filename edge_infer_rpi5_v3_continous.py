import socket
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from shard_utils import split_model

model = AutoModelForCausalLM.from_pretrained("distilgpt2")
(wte, wpe, drop, part1_blocks), _, _ = split_model(model, cut_layer_idx=3)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
for block in part1_blocks:
    block.eval().to(dtype=torch.float32)

text = "Tomorrow we will go"
input_ids = tokenizer.encode(text, return_tensors="pt").to(torch.long)

HOST = '192.168.1.45'
PORT = 5005

for step in range(10):
    with torch.no_grad():
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        inputs_embeds = wte(input_ids) + wpe(position_ids)
        hidden_state = drop(inputs_embeds)
        for block in part1_blocks:
            hidden_state = block(hidden_state)[0]

    data = hidden_state.numpy().astype("float32").tobytes()
    print(f"\n[{step+1}] Sending hidden_state of shape: {hidden_state.shape} ({len(data)} bytes)")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(data)
        s.shutdown(socket.SHUT_WR)
        print("Sent hidden_state, waiting for response...")
        token_id_bytes = s.recv(4)
        next_token_id = int.from_bytes(token_id_bytes, byteorder='big')

    input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)
    print(f"[{step+1}] Generated so far: {tokenizer.decode(input_ids[0])}")

import socket
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from shard_utils import split_model

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
_, (part2_blocks, ln_f), lm_head = split_model(model, cut_layer_idx=3)

for block in part2_blocks:
    block.eval().to(dtype=torch.float32)
ln_f.eval().to(dtype=torch.float32)
lm_head.eval().to(dtype=torch.float32)

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Server config
HOST = '0.0.0.0'
PORT = 5002

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)
    print(f"Waiting for incoming connections on port {PORT}...")

    while True:
        conn, addr = server_socket.accept()
        with conn:
            print(f"\nConnected by {addr}")
            data = b''
            while True:
                packet = conn.recv(4096)
                if not packet:
                    break
                data += packet

        # Convert bytes → tensor
        hidden_dim = 768
        try:
            seq_len = len(data) // (4 * hidden_dim)
            hidden = np.frombuffer(data, dtype=np.float32).reshape(1, seq_len, hidden_dim)
            hidden_tensor = torch.tensor(hidden, dtype=torch.float32)

            # Run shard 2
            with torch.no_grad():
                h = hidden_tensor
                for block in part2_blocks:
                    h = block(h)[0]
                h = ln_f(h)
                logits = lm_head(h)
                predicted_id = torch.argmax(logits[:, -1, :], dim=-1)

            # Decode token
            generated_text = tokenizer.decode(predicted_id)
            print(f"Final output: {generated_text}")

        except Exception as e:
            print(f"Error processing tensor: {e}")

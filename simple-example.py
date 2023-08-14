from llama.mpmodel import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
import torch

torch.set_default_tensor_type(torch.cuda.HalfTensor)
tokenizer = Tokenizer("tokenizer.model")
args = ModelArgs()
args.vocab_size = tokenizer.n_words
# args.n_layers = 1
args.max_seq_len=32

model = Transformer(args)

checkpoint = torch.load('llama-2-7b-chat/consolidated.00.pth',map_location='cpu')
del checkpoint['rope.freqs']

model.load_state_dict(checkpoint)

from llama.generation import Llama
l = Llama(model,tokenizer)
# l.text_completion("hello,")


prompts = [
    # For these prompts, the expected answer is the natural continuation of the prompt
    "I believe the meaning of life is",
    "Simply put, the theory of relativity states that "
]
results = l.text_completion(
    prompts,
    max_gen_len=32,
    temperature=0.6,
    top_p=0.9,
)
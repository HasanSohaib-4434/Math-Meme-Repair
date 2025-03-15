import streamlit as st
import torch
from deepseke_r1 import DeepSeKeR1Tokenizer, DeepSeKeR1ForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset

math_memes = [
    {"incorrect": "8 ÷ 2(2+2) = 1", "correct": "8 ÷ 2(2+2) = 16 because according to PEMDAS, we first solve inside the parentheses (2+2=4), then proceed left to right: 8 ÷ 2 × 4 = 4 × 4 = 16."},
    {"incorrect": "0.999... = 1 is false", "correct": "0.999... = 1 is true because repeating decimals can be expressed as limits, and mathematically, they are equivalent."}
]

def train_model():
    dataset = Dataset.from_dict({"incorrect": [m["incorrect"] for m in math_memes], "correct": [m["correct"] for m in math_memes]})
    tokenizer = DeepSeKeR1Tokenizer.from_pretrained("deepseke-r1")
    
    def tokenize_function(examples):
        return tokenizer(examples["incorrect"], text_target=examples["correct"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    model = DeepSeKeR1ForSeq2SeqLM.from_pretrained("deepseke-r1")
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets
    )
    trainer.train()
    model.save_pretrained("./math_meme_corrector")
    tokenizer.save_pretrained("./math_meme_corrector")
    return model, tokenizer

def correct_math_meme(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True)

st.title("Math Meme Corrector")
if st.button("Fine-Tune Model"):
    model, tokenizer = train_model()
    st.success("Model fine-tuned successfully!")

prompt = st.text_input("Enter an incorrect math meme:")
if st.button("Correct Meme") and prompt:
    model = DeepSeKeR1ForSeq2SeqLM.from_pretrained("./math_meme_corrector")
    tokenizer = DeepSeKeR1Tokenizer.from_pretrained("./math_meme_corrector")
    correction = correct_math_meme(prompt, model, tokenizer)
    st.write(correction)

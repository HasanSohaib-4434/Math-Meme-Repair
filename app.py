import json

data = [
  {
    "incorrect": "8 ÷ 2(2+2) = 1",
    "correct": "16",
    "explanation": "PEMDAS states that division and multiplication have equal precedence. Left to right: 8 ÷ 2 × 4 = 4 × 4 = 16."
  },
   {
    "incorrect": "8 ÷ 2*(2+2) = 1",
    "correct": "16",
    "explanation": "PEMDAS states that division and multiplication have equal precedence. Left to right: 8 ÷ 2 × 4 = 4 × 4 = 16."
  },
   {
    "incorrect": "8 / 2(2+2) = 1",
    "correct": "16",
    "explanation": "PEMDAS states that division and multiplication have equal precedence. Left to right: 8 ÷ 2 × 4 = 4 × 4 = 16."
  },
   {
    "incorrect": "8 / 2*(2+2) = 1",
    "correct": "16",
    "explanation": "PEMDAS states that division and multiplication have equal precedence. Left to right: 8 ÷ 2 × 4 = 4 × 4 = 16."
  },
  {
    "incorrect": "1 + 1 × 0 = 0",
    "correct": "1",
    "explanation": "Multiplication comes before addition: 1 × 0 = 0, then 1 + 0 = 1."
  },
  {
    "incorrect": "6 ÷ 2(3) = 1",
    "correct": "9",
    "explanation": "Following PEMDAS: 6 ÷ 2 × 3 = 3 × 3 = 9."
  },
  {
    "incorrect": "6 ÷ 2*(3) = 1",
    "correct": "9",
    "explanation": "Following PEMDAS: 6 ÷ 2 × 3 = 3 × 3 = 9."
  },
  {
    "incorrect": "50 - 5 × 10 = 450",
    "correct": "0",
    "explanation": "Multiplication first: 5 × 10 = 50, then 50 - 50 = 0."
  },
  {
    "incorrect": "100 ÷ 5 × 2 = 5",
    "correct": "40",
    "explanation": "Division and multiplication are equal precedence; solve left to right: 100 ÷ 5 = 20, then 20 × 2 = 40."
  },
  {
    "incorrect": "7 + 3 × 0 = 7 + 3 = 10",
    "correct": "7",
    "explanation": "Multiplication first: 3 × 0 = 0, then 7 + 0 = 7."
  },
  {
    "incorrect": "12 ÷ 4 + 2 = 2",
    "correct": "5",
    "explanation": "Division first: 12 ÷ 4 = 3, then 3 + 2 = 5."
  },
  {
    "incorrect": "15 - 3 × 4 = 48",
    "correct": "3",
    "explanation": "Multiplication first: 3 × 4 = 12, then 15 - 12 = 3."
  },
  {
    "incorrect": "5 + 5 × 5 = 50",
    "correct": "30",
    "explanation": "Multiplication first: 5 × 5 = 25, then 5 + 25 = 30."
  },
  {
    "incorrect": "10 ÷ 2 × 5 = 1",
    "correct": "25",
    "explanation": "Left to right: 10 ÷ 2 = 5, then 5 × 5 = 25."
  },
  {
    "incorrect": "2*(5+5) ÷ 2 = 2",
    "correct": "10",
    "explanation": "Parentheses first: 5 + 5 = 10, then 2 × 10 = 20, finally 20 ÷ 2 = 10."
  },
   {
    "incorrect": "2*(5+5) ÷ 2 = 2",
    "correct": "10",
    "explanation": "Parentheses first: 5 + 5 = 10, then 2 × 10 = 20, finally 20 ÷ 2 = 10."
  },
  {
    "incorrect": "18 ÷ 3 × 2 = 3",
    "correct": "12",
    "explanation": "Division and multiplication have equal precedence: 18 ÷ 3 = 6, then 6 × 2 = 12."
  },
  {
    "incorrect": "3 + 3 × 3 + 3 = 21",
    "correct": "15",
    "explanation": "Multiplication first: 3 × 3 = 9, then 3 + 9 + 3 = 15."
  },
  {
    "incorrect": "100 ÷ 10 × 10 = 1",
    "correct": "100",
    "explanation": "Left to right: 100 ÷ 10 = 10, then 10 × 10 = 100."
  },
  {
    "incorrect": "(6 + 4) ÷ 2 = 10",
    "correct": "5",
    "explanation": "Parentheses first: 6 + 4 = 10, then 10 ÷ 2 = 5."
  },
  {
    "incorrect": "9 - 3 ÷ 3 = 2",
    "correct": "8",
    "explanation": "Division first: 3 ÷ 3 = 1, then 9 - 1 = 8."
  },
  {
    "incorrect": "14 - 2(3) = 36",
    "correct": "8",
    "explanation": "Multiplication first: 2 × 3 = 6, then 14 - 6 = 8."
  },
  {
    "incorrect": "25 ÷ 5 × 2 = 5",
    "correct": "10",
    "explanation": "Left to right: 25 ÷ 5 = 5, then 5 × 2 = 10."
  },
  {
    "incorrect": "4(2+3) = 4 × 2 + 3",
    "correct": "20",
    "explanation": "Parentheses first: 2 + 3 = 5, then 4 × 5 = 20."
  }
]


with open("math_meme_correction.json", "w") as f:
    json.dump(data, f, indent=4)
!pip install -q unsloth
!pip install -q bitsandbytes transformers accelerate peft trl datasets huggingface_hub
import os
os.environ["HF_TOKEN"] =# simply paste you hugging face token here


from huggingface_hub import whoami
print(whoami())from unsloth import FastLanguageModel
import torch

model_id = "deepseek-ai/deepseek-math-7b-instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_id,
    max_seq_length=2048, 
    dtype=torch.float16,
    load_in_4bit=True 
)

tokenizer.pad_token = tokenizer.eos_token
from transformers import AutoTokenizer


model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",   
    use_gradient_checkpointing = "unsloth",
    random_state = 3977,
    use_rslora = False,  
    loftq_config = None, 
)from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./deepseek-math-fine",
    per_device_train_batch_size=1,  
    gradient_accumulation_steps=4,  
    optim="adamw_8bit", 
    save_steps=500,
    eval_strategy="steps", 
    eval_steps=500,
    logging_steps=100,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,  
    num_train_epochs=7,
    push_to_hub=False,
    save_total_limit=2, 
    report_to="none", 
    run_name="deepseek-math-finetune"
)
from datasets import load_dataset

dataset = load_dataset("json", data_files="math_meme_correction.json")

EOS_TOKEN = tokenizer.eos_token 

def formatting_prompts_func(examples):
    texts = [inc + " -> " + cor + EOS_TOKEN for inc, cor in zip(examples["incorrect"], examples["correct"])]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

def tokenize_function(examples):
    model_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=2048)
    model_inputs["labels"] = model_inputs["input_ids"].copy()  
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(tokenized_datasets)
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["train"],  
    tokenizer=tokenizer, 
)

trainer.train()
import torch

def test_model(model, tokenizer, test_cases):
    model.eval() 
    results = []

    for test in test_cases:
        input_text = test + " ->"
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)

        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append((test, corrected_text))

    return results

test_cases = [
    "2 + 2 = 5",
    "The square root of 25 is 6",
    "7 * 8 = 54",
    "The derivative of x^2 is x",
]

results = test_model(model, tokenizer, test_cases)

for incorrect, corrected in results:
    print(f"Incorrect: {incorrect}")
    print(f"Corrected: {corrected}\n")


 



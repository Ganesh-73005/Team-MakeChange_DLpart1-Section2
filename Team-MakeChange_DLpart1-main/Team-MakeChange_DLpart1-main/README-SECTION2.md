WikiHow Heading Generation
This repository contains the code for a heading generation model for WikiHow articles.

The model is trained using a fine-tuned LED (Language Model for Dialogue Applications) model from Hugging Face.

Requirements
To use the code in this repository, you will need the following packages installed:

transformers
datasets
pandas
rouge_score
You can install these packages using pip:

bash

Verify

Open In Editor
Edit
Copy code
!pip install transformers datasets pandas rouge_score
Data
The dataset used for training the model is a CSV file containing WikiHow articles. The CSV file contains the following columns:

Article Title: The title of the WikiHow article.
Subheading: The heading of the section of the WikiHow article.
Paragraph: The text of the paragraph.
Usage
The code in this repository can be used to:

Train a heading generation model: The code includes a function for training the model.
Generate headings for new paragraphs: The code includes a function for generating headings for new paragraphs.
Running the code
To run the code, you can execute the following commands in a Jupyter notebook:

python

Verify

Open In Editor
Edit
Copy code
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from transformers import LEDTokenizer, LEDForConditionalGeneration
import torch
from datasets import load_metric
from transformers import AutoModelForSeq2SeqLM
import numpy as np
from datasets import Dataset, load_metric, load_dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import transformers

# Load the csv file
file_path = './wikiHow.csv'
df = pd.read_csv(file_path)

# ... (rest of the code) ...

# Train the model
trainer.train()

# Generate headings for new paragraphs
sample_paragraph = "Virat kohli is an inspiration to many people around the world"
data = [sample_paragraph]
df = pd.DataFrame(data, columns=['Paragraph'])
df["Paragraph"][0]
df_test = Dataset.from_pandas(df)
df_test

tokenizer = LEDTokenizer.from_pretrained("/content/checkpoint-60")
model = LEDForConditionalGeneration.from_pretrained("/content/checkpoint-60").to("cuda").half()

def generate_answer(batch):
  inputs_dict = tokenizer(batch["Paragraph"], padding="max_length", max_length=512, return_tensors="pt", truncation=True)
  input_ids = inputs_dict.input_ids.to("cuda")
  attention_mask = inputs_dict.attention_mask.to("cuda")
  global_attention_mask = torch.zeros_like(attention_mask)
 
  predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
  batch["generated_heading"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
  return batch

result = df_test.map(generate_answer, batched=True, batch_size=2)

print(result["generated_heading"])
Evaluation
The model is evaluated using the ROUGE metric. The ROUGE metric is a standard metric for evaluating the quality of text summaries.

The code includes a function for computing the [Rouge](https://www.freecodecamp.org/news/what-is-rouge-and-how-it-works-for-evaluation-of-summaries-e059fb8ac840/) score.

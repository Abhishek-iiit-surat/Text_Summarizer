

from transformers import pipeline, set_seed
from transformers import pipeline
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
from datasets import load_dataset, load_metric

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import nltk
from nltk.tokenize import sent_tokenize

from tqdm import tqdm
import torch

nltk.download("punkt")


device = "cuda" if torch.cuda.is_available() else "cpu"

model_ckpt = "google/pegasus-cnn_dailymail"

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

# Load data
dataset_samsum = load_dataset("samsum")

dialogue=dataset_samsum['test'][0]['dialogue']

pipe=pipeline('summarization',model=model_ckpt)
pipe_out=pipe(dialogue)
pipe_out

def generate_batch_sized_chunks(list_of_elements, batch_size):
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i:i+batch_size]

def calculate_metric_on_test_ds(dataset, metric, model, tokenizer, batch_size=16, device=device, column_text="article", column_summary="highlights"):
    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):

        inputs = tokenizer(article_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")

        summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                                   attention_mask=inputs["attention_mask"].to(device),
                                   length_penalty=0.8, num_beams=8, max_length=128)

        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                             for s in summaries]

        decoded_summaries = [d.replace("", " ") for d in decoded_summaries]

        metric.add_batch(predictions=decoded_summaries, references=target_batch)

    score = metric.compute()
    return score

rouge_metric = load_metric('rouge')

score = calculate_metric_on_test_ds(dataset_samsum['test'], rouge_metric, model_pegasus, tokenizer, column_text='dialogue',column_summary='summary',batch_size=8)

rouge_names=["rouge1","rouge2","rougeL","rougeLsum"]
rouge_dict = dict((rn,score[rn].mid.fmeasure) for rn in rouge_names)

pd.DataFrame(rouge_dict,index=['pegasus'])

def convert_examples_to_features(example_batch):
  input_encodings = tokenizer(example_batch['dialogue'],max_length=1024,truncation=True)

  with tokenizer.as_target_tokenizer():
    target_encodings = tokenizer(example_batch['summary'], max_length=128,truncation=True)

    return{
        'input_ids':input_encodings['input_ids'],
        'attention_mask':input_encodings['attention_mask'],
        'labels':target_encodings['input_ids']
    }

dataset_samsum_pt=dataset_samsum.map(convert_examples_to_features,batched=True)

dataset_samsum_pt['train'][0]

from transformers import DataCollatorForSeq2Seq

seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)



from transformers import TrainingArguments, Trainer


trainer_args = TrainingArguments(
    output_dir='pegasus-samsum',num_train_epochs=1,warmup_steps=500,
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    weight_decay=0.01,logging_steps=10,
    evaluation_strategy='steps',eval_steps=500,save_steps=1e6,
    gradient_accumulation_steps=16
)

trainer = Trainer(model=model_pegasus,args=trainer_args,
                  tokenizer=tokenizer,data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt["train"],
                  eval_dataset=dataset_samsum_pt["validation"])

trainer.train()

score=calculate_metric_on_test_ds(
    dataset_samsum['test'],rouge_metric, trainer.model,tokenizer,batch_size=2,column_text='dialogue',column_summary='summary'
)
rouge_dict=dict((rn,score[rn].mid.fmeasure)for rn in rouge_names)
pd.DataFrame(rouge_dict,index=[f'pegasus'])

# save model
model_pegasus.save_pretrained("pegasus-samsum-model")

tokenizer.save_pretrained("tokenizer")

score=calculate_metric_on_test_ds(
    dataset_samsum['test'],rouge_metric,trainer.model,tokenizer,batch_size=2,column_text='dialogue',column_summary='summary')

rouge_dict=dict((rn,score[rn].mid.fmeasure) for rn in rouge_names)
pd.DataFrame(rouge_dict,index=[f'pegasus'])

tokenizer = AutoTokenizer.from_pretrained("tokenizer")

dataset_samsum=load_dataset("samsum")

# sample_text=dataset_samsum["test"][0]["dialogue"]
sample_text="Person A (Ram): Hey, Shyam! Long time no talk, how's life?Person B (Shyam): Ram, my friend! Life's been quite the rollercoaster. How about you?Person A (Ram): Oh, you know, the usual. Work, family, and a bit of adventure. We should meet up soon, share stories.Person B (Shyam): Absolutely! How about next weekend? Coffee and reminiscing?Person A (Ram): Perfect! Looking forward to it, Shyam.Person B (Shyam): Likewise, Ram. Take care till then!"
# reference=dataset_samsum["test"][0]["summary"]

sample_text

reference

gen_kwargs={"length_penalty":0.8,"num_beams":8,"max_length":128}
pipe=pipeline("summarization",model="pegasus-samsum-model",tokenizer=tokenizer)



print("\nModel Summary:")
print(pipe(sample_text, **gen_kwargs)[0]["summary_text"])
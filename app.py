from flask import Flask, render_template, request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

app = Flask(__name__)

# Load the trained model and tokenizer
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained('pegasus-samsum-model/pytorch_model.bin')  # Update with the actual path
tokenizer = AutoTokenizer.from_pretrained('tokenizer')  # Update with the actual path

# Create a pipeline for summarization
pipe = pipeline("summarization", model=model_pegasus, tokenizer=tokenizer)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}
        summary = pipe(input_text, **gen_kwargs)[0]["summary_text"]
        return render_template('index.html', input_text=input_text, summary=summary)
    return render_template('index.html', input_text='', summary='')

if __name__ == '__main__':
    app.run(debug=True)

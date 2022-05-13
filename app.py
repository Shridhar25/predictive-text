# save this as app.py
from asyncio.windows_events import NULL
from ensurepip import bootstrap
from operator import index
from flask import Flask, jsonify, redirect, render_template, url_for, request


import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer




app = Flask(__name__)


@app.route("/")  
def home():
    return render_template("login.html")

@app.route("/predict", methods=["POST" , "GET"])
def predict():
    if request.method == "POST":
        word1 = request.form["text"]
        output_word = ""
        last_word = ""

        list1 = ['!' , '@' , '#' , '$' , '%' , '^' , '&' , '*' , '(' , ')' , ';', ':' , ',' , '"' , '?' , '/' , '<' , '>' ,'.' , '[' ,']' , '{' , '}' , '|' , '_' , '-' , '+' , '=' , '1' , '2' , '3' , '4' , '5' , '6' , '7' , '8' , '9' , '0']
        if word1 == "":
            last_word = "Oops! You forgot to enter text...😕"
        elif word1[-1] in list1:
            last_word = "Invalid Input 😕"
            output_word = word1
        else:
            next_word = 6
            data = open("Metamorphosis.txt", encoding="utf8").read()
            tokenizer = Tokenizer()
            corpus = data.lower().split("\n")
            tokenizer.fit_on_texts(corpus)

            input_sequences = []
            for line in corpus:
                token_list = tokenizer.texts_to_sequences([line])[0]
                for i in range(1, len(token_list)):
                    n_gram_sequence = token_list[:i + 1]
                    input_sequences.append(n_gram_sequence)

            max_sequence_len = max([len(x) for x in input_sequences])

            input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
            json_file = open('model_lstm_bilstm_combo.json','r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("model_lstm_bilstm_combo.h5")

            model = loaded_model

            for i in range(next_word):
                token_list = tokenizer.texts_to_sequences([word1])[0]
                token_list = (pad_sequences([token_list], maxlen=max_sequence_len - 1,padding="pre"))
                predicted = np.argmax(model.predict(token_list), axis=-1)
                
                for word, index in tokenizer.word_index.items():
                    if (index == predicted):
                        last_word = word
                        output_word = word1 + " " + word
                        break


        return render_template("index.html", pred=output_word,last_word=last_word)

    else:
        return render_template("index.html")



if __name__ == '__main__':
    app.run(debug=True)
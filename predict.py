from __future__ import print_function
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import ai_integration
import sys
import os

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000

def main():
    model = load_model('model.h5')
    model.load_weights("weights.h5")

    while True:
        with ai_integration.get_next_input(inputs_schema={"text": {"type": "text"}}) as inputs_dict:
        # If an exception happens in this 'with' block, it will be sent back to the ai_integration library
            X_raw = [inputs_dict("text")]         

            X, word_index = tokenize_data(X_raw)

            predictions = model.predict(x=X, batch_size=1)

            is_positive = predictions[0][1] >= 0.5
            status_txt = "Positive" if is_positive else "Negative"
            # This model was initially handling a set of inputs, yet now it only receives one: what implications could this have?
            # What is being done to X, and why?
            result_data = {
                "content-type": 'text/plain',
                "data": "Fake output",
                "success": True
            }
            ai_integration.send_result(result_data)

def tokenize_data(X_raw):
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(X_raw)
    sequences = tokenizer.texts_to_sequences(X_raw)
    word_index = tokenizer.word_index
    X_processed = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return X_processed, word_index

"""
cd ~/.ssh/interns
cd ~/.ssh
ls
chmod 700 ~/.ssh/interns
ssh -i ~/.ssh/interns ubuntu@intern1.deepai.org
"""
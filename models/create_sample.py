import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalAveragePooling1D
import numpy as np
import string
import sys

# 1. Character-level vocabulary
chars = list(string.ascii_lowercase + string.digits + " ")
char2idx = {ch: i + 1 for i, ch in enumerate(chars)}  # reserve 0 for padding
vocab_size = len(char2idx) + 1

# 2. Preprocessing function


def preprocess(text, maxlen=20):
    text = text.lower()
    seq = [char2idx.get(ch, 0) for ch in text]
    seq = seq[:maxlen] + [0]*(maxlen - len(seq))
    return np.array(seq)

# 3. Model definition


def create_model(maxlen=20):
    inp = Input(shape=(maxlen,))
    x = Embedding(vocab_size, 16)(inp)
    x = LSTM(32, return_sequences=True)(x)
    x = GlobalAveragePooling1D()(x)
    out = Dense(2, activation='softmax')(x)  # %string, %number
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model


model = create_model()

# 4. Example training data
examples = [
    ("hello", [1.0, 0.0]),
    ("", [0.0, 0.0]),
    ("42", [0.0, 1.0]),
    ("h3110", [0.6, 0.4]),
    ("123", [0.0, 1.0]),
    ("a1b2", [0.5, 0.5]),
    ("world2025", [0.5, 0.5]),
    ("sam1", [0.75, 0.25]),
]

X = np.array([preprocess(text) for text, _ in examples])
y = np.array([label for _, label in examples])

# 5. Train model
model.fit(X, y, epochs=50, verbose=0)

# 6. Test prediction

def analyze_text(text):
    input_seq = preprocess(text)
    pred = model.predict(np.expand_dims(input_seq, axis=0), verbose=0)[0]
    
    print(pred)
    
    return {
        "input": text,
        "percent_string": round(pred[0]*100, 2),
        "percent_number": round(pred[1]*100, 2)
    }


# Test
# print(analyze_text("Hello"))
# print(analyze_text("h3110"))
# print(analyze_text("123456"))
# print(analyze_text("AI2025"))
# print(analyze_text("A"))

model.save("string_number_classifier.keras")

sys.modules[__name__] = analyze_text
# __all__ = [analyze_text, preprocess]
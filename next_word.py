import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import os
import re
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DICTIONARY_PATH = "./dictionary.txt"
DICTIONARY_list =  np.loadtxt(DICTIONARY_PATH, dtype= str)
DICTIONARY_dict = {word: idx for idx, word in enumerate(DICTIONARY_list)}
SOS_IDX = DICTIONARY_dict["<sos>"]
EOS_IDX = DICTIONARY_dict["<eos>"]
UNK_IDX = DICTIONARY_dict["<unk>"]
print("SOS_IDX:",SOS_IDX,"EOS_IDX:",EOS_IDX,"UNK_IDX:", UNK_IDX)

#LSTM WITHOUT ATTENTION
class BiLSTM_N_gramModel_WITHOUT_ATTENTION(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(BiLSTM_N_gramModel_WITHOUT_ATTENTION, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Sequential(nn.Linear(hidden_size * 2, embedding_dim),
                                      nn.Linear(embedding_dim, vocab_size),)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.bilstm(embedded)
        logits = self.fc(output)
        return logits
    
vocab_size = len(DICTIONARY_list)
embedding_dim = 300
hidden_size = 256


model_WITHOUT_ATTENTION = BiLSTM_N_gramModel_WITHOUT_ATTENTION(vocab_size, embedding_dim, hidden_size).to(DEVICE)
model_WITHOUT_ATTENTION = torch.quantization.quantize_dynamic(
    model_WITHOUT_ATTENTION,
    # qconfig_spec={nn.Embedding},  # Specify which submodules to quantize
    dtype=torch.qint8
)

state_dict = torch.load("./LSTM1_checkpoint.pth",map_location=torch.device('cpu'))
model_WITHOUT_ATTENTION.load_state_dict(state_dict["model_state_dict"])
# print(model)

def predict_next_word_without_attention(model, context, vocab_dict, index_dict, top_k = 3):
    model.eval()
    
    context = re.sub(r"['’]", "' ", context)
    context = re.sub(r"[^a-zA-Z'’ ]", ' ', context)
    context = ["<sos>"] + context.lower().split()

    with torch.no_grad():
        # Convert context to indices
        context_indices = [vocab_dict.get(word, UNK_IDX) for word in context]
        input_tensor = torch.LongTensor(context_indices).unsqueeze(0)

        # Get model prediction
        output = model(input_tensor.to(DEVICE))
        # output = output[0]
        # print("*_*"*9)
        # print("output:\n",output)
        # print("*_*"*9)
        predicted_index = torch.argmax(output[0, -1, :]).item()

        # Convert index to word
        predicted_word = index_dict.get(predicted_index, "<unk>")

        top_values, top_indices = torch.topk(torch.nn.Softmax(dim=0)(output[0, -1, :]), k=top_k)

        top_words = [index_dict.get(top_word, "<unk>") for top_word in top_indices.tolist()]


        top_k_dict = dict(zip(top_words, top_values.tolist()))

        # print(f"Top {top_k} predicted words:\n", top_k_dict)

        return top_k_dict

#END LSTM WITHOUT ATTENTION




# MODEL
class BiLSTM_N_gramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(BiLSTM_N_gramModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim = hidden_size*2, num_heads=1, dropout=0.0,
                                               batch_first=True)
        self.fc = torch.nn.Sequential(nn.Linear(hidden_size * 2, embedding_dim),
                                      nn.Linear(embedding_dim, vocab_size),)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.bilstm(embedded)
        attn_output , attn_output_weights = self.attention(query=output, key=output, value=output,
                                                           need_weights=True, average_attn_weights=True)
        logits = self.fc(output)
        return logits, attn_output_weights

# Initialize the model


model = BiLSTM_N_gramModel(vocab_size, embedding_dim, hidden_size).to(DEVICE)
model = torch.quantization.quantize_dynamic(
    model,
    # qconfig_spec={nn.Embedding},  # Specify which submodules to quantize
    dtype=torch.qint8
)

state_dict = torch.load("./LSTM_attention_checkpoint.pth",map_location=torch.device('cpu'))
model.load_state_dict(state_dict["model_state_dict"])
print(model)

def predict_next_word(model, context, vocab_dict, index_dict, top_k = 3):
    model.eval()
    
    context = re.sub(r"['’]", "' ", context)
    context = re.sub(r"[^a-zA-Z'’ ]", ' ', context)
    context = ["<sos>"] + context.lower().split()

    with torch.no_grad():
        # Convert context to indices
        context_indices = [vocab_dict.get(word, UNK_IDX) for word in context]
        input_tensor = torch.LongTensor(context_indices).unsqueeze(0)

        # Get model prediction
        output = model(input_tensor.to(DEVICE))
        output = output[0]
        # print("*_*"*9)
        # print("output:\n",output)
        # print("*_*"*9)
        predicted_index = torch.argmax(output[0, -1, :]).item()

        # Convert index to word
        predicted_word = index_dict.get(predicted_index, "<unk>")

        top_values, top_indices = torch.topk(torch.nn.Softmax(dim=0)(output[0, -1, :]), k=top_k)

        top_words = [index_dict.get(top_word, "<unk>") for top_word in top_indices.tolist()]


        top_k_dict = dict(zip(top_words, top_values.tolist()))

        # print(f"Top {top_k} predicted words:\n", top_k_dict)

        return top_k_dict


ngram_to_index = DICTIONARY_dict
index_to_ngram = {value: key for key, value in DICTIONARY_dict.items()}
# input_text = "Ubu bufatanye buratwongerera imbaraga mu"
# predicted_word = predict_next_word(model, input_text, ngram_to_index, index_to_ngram, top_k= 4)
print("=============================================")
# print(f'Next word prediction: {predicted_word}')
print("=============================================")


# st.title("Kinyarwanda Auto-Complete Application")
# Inject custom CSS with your desired font size
st.markdown("""
    <style>
    .title {
        font-size:35px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply the custom CSS class to your title
st.markdown('<p class="title"><strong>Kinyarwanda Auto-Complete Application</strong></p>', unsafe_allow_html=True)

st.sidebar.text("Next Word Prediction")

# Initialize the session state for the text area if it doesn't exist
if 'input_text' not in st.session_state:
    st.session_state['input_text'] = " "

input_text = st.text_area("Enter your text here", value=st.session_state['input_text'])
# st.text_area("Predicted List is Here",key="predicted_list")

top_k = st.sidebar.slider("How many words do you need", 1 , 25, 1) #some times it is possible to have less words
user_k = int(top_k)
model_name = st.sidebar.selectbox(label='Select Model to Apply',  options=['BiLSTM', 'BiLSTM + ATTENTION'], index=0,  key = "model_name")
# Display text based on the selected model
if model_name == 'BiLSTM':
    predicted_word = predict_next_word_without_attention(model_WITHOUT_ATTENTION, input_text, ngram_to_index, index_to_ngram, top_k= user_k)
    number_of_buttons = user_k
    # Create a list of button labels
    button_labels = [f'{list(predicted_word.keys())[i]} : {"{:.04f}".format(list(predicted_word.values())[i])}' for i in range(number_of_buttons)]
    # Generate a row of columns with buttons
    cols = st.columns(number_of_buttons)
    for i, col in enumerate(cols):
        with col:
            if st.button(button_labels[i], key=f'button_{i}'):
                st.session_state['input_text'] = f"{input_text} {button_labels[i].split()[0]}"
                # st.write(f'You clicked {button_labels[i]}')
elif model_name == 'BiLSTM + ATTENTION':
    predicted_word = predict_next_word(model, input_text, ngram_to_index, index_to_ngram, top_k= user_k)
    # st.write("BILSTM with Attention Next word prediction:",predicted_word)
    # st.write('You have selected the BiLSTM model with Attention.')
    # Number of buttons you want to display
    number_of_buttons = user_k

    # Create a list of button labels
    button_labels = [f'{list(predicted_word.keys())[i]} {list(predicted_word.values())[i]}' for i in range(number_of_buttons)]

    # Generate a row of columns with buttons
    cols = st.columns(number_of_buttons)
    for i, col in enumerate(cols):
        with col:
            if st.button(button_labels[i], key=f'button_{i}'):
                st.session_state['input_text'] = f"{input_text} {button_labels[i].split()[0]}"
                # st.write(f'You clicked {button_labels[i]}')


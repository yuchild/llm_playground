import streamlit as st
from transformers import pipeline

# Load the Hugging Face model
@st.cache_resource
def load_pipeline():
    model_id = "meta-llama/Llama-3.3-70B-Instruct"  # Replace with your desired Hugging Face model
    return pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": "bfloat16"},
        device_map="auto",
    )

# Streamlit app configuration
st.set_page_config(
    page_title="Pirate Chatbot",
    page_icon="☠️",
    layout="centered",
)

st.title("☠️ Pirate Chatbot ☠️")
st.write("Talk to the pirate chatbot in pirate speak!")

# Initialize the NLP pipeline
nlp_pipeline = load_pipeline()

# Input box for user queries
user_input = st.text_input("Ask a question:", placeholder="Type your question here...")

# Display system message
if user_input:
    with st.spinner("Thinking like a pirate..."):
        # System message template
        messages = [
            {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
            {"role": "user", "content": user_input},
        ]
        
        # Generate response
        try:
            outputs = nlp_pipeline(
                messages,
                max_new_tokens=256,
            )
            response = outputs[0]["generated_text"]
            st.success(response)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

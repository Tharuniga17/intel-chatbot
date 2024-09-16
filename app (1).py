import streamlit as st
from transformers import pipeline

# Load the pipeline with your fine-tuned LLama2 model
pipe = pipeline("text-generation", model="KishoreYuva/finetuned_llama-2_using-platypus")

# Function to generate response
def generate_response(prompt):
    # Create the instruction from the prompt
    instruction = f"### Instruction:\n{prompt}\n\n### Response:\n"

    # Generate the response using the pipeline
    result = pipe(instruction, max_new_tokens=40)
    
    # Extract and return the generated text excluding the instruction part
    response = result[0]['generated_text'][len(instruction):]
    return response

# Streamlit UI
def main():
    st.title("Chat with LLama2")
    st.write("Welcome to the LLama2 chat interface! Ask a question or provide a prompt.")

    # User input for prompt
    prompt = st.text_input("You:", "")

    if st.button("Ask"):
        if prompt:
            st.write("LLama2 is typing...")
            response = generate_response(prompt)
            st.write("LLama2:", response)
        else:
            st.write("Please enter a prompt to continue.")

if __name__ == "__main__":
    main()

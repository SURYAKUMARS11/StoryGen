import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set up Streamlit UI
st.set_page_config(page_title="Story Generation with GPT-2", page_icon="✨")
st.title("📖 StoryGen")

# User input for prompt
user_prompt = st.text_area("✏️ Enter your prompt:")

# Genre selection
genres = ["General", "Horror", "Sci-Fi", "Educational", "Romance", "Comedy", "Adventure"]
selected_genre = st.selectbox("🎭 Select Genre:", genres)

# Button to trigger generation
generate_button = st.button("Generate Story 🚀")

# Output area to display generated story
output_area = st.empty()

# Function to generate story on button click
def generate_story():
    if not user_prompt:
        st.warning("⚠️ Prompt cannot be empty. Please enter a prompt.")
        return

    genre_prefix = {
        "General": "",
        "Horror": "Once upon a dark and stormy night, ",
        "Sci-Fi": "In a galaxy far, far away, ",
        "Educational": "In the realm of knowledge, ",
        "Romance": "In the tender embrace of love, ",
        "Comedy": "In the land of laughter, ",
        "Adventure": "Embarking on a thrilling journey, "
    }

    selected_prefix = genre_prefix.get(selected_genre, "")

    with st.spinner("🔍 Generating..."):
        encoded_user_prompt = tokenizer.encode(selected_prefix + user_prompt, add_special_tokens=True, return_tensors="pt")

        output_sequences = model.generate(
            input_ids=encoded_user_prompt,
            max_length=300,
            temperature=0.8,
            top_k=30,
            top_p=0.9,
            repetition_penalty=1.0,
            do_sample=True,
            num_return_sequences=1
        )

        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_story = ""
        for generated_sequence in output_sequences:
            generated_sequence = generated_sequence.tolist()
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
            text = text[: text.find(tokenizer.eos_token)]
            generated_story += text

        # Display generated story without a box
        st.markdown(f"**📜 Generated Story:**\n{generated_story}")

# Event handler for button click
if generate_button:
    generate_story()

# Add some additional styling
st.sidebar.title("About")
st.sidebar.info(
    "This app uses GPT-2 to generate creative stories based on user prompts and genres. "
    "Adjust the prompt, select a genre, and click 'Generate Story' to see the magic happen! ✨"
)

# Footer with Quadra Squad
st.markdown(
    """
    \n\n
    [![Quadra Squad](https://img.shields.io/badge/Quadra%20Squad-Transforming%20Ideas-1F425F?style=flat-square&labelColor=1F425F&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAVklEQVR42mNkAAIAAAoAAv/lxKUAAAAASUVORK5CYII=)](https://www.quadrasquad.com/)
    """
)

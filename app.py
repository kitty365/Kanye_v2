import gradio as gr
from PIL import Image
from kanye_base_code import retrieve_with_validation, build_prompt, client  # from your own module

# Main interface function
def kanye_rag_interface(user_query):
    chunks, is_valid = retrieve_with_validation(user_query, k=5)
    if not is_valid:
        return "This question does not seem relevant to the Kanye context."

    prompt = build_prompt(chunks, user_query)
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant focused on culture, psychology & art."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Build UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Column():
        gr.Image("kanye.jpeg", show_label=False, container=False, elem_id="kanye-img")

        gr.Markdown("##  **KanyeRAG** ")
        gr.Markdown("This project aims to interpret the thoughts, lyrics, tweets and emotional patterns of Kanye West using a Retrieval Augmented Generation system. By applying semantic search and clustering across diverse sources such as tweets, lyrics, biographical texts and academic research, the system seeks to offer a philosophical perspective on Kanye’s inner world")
    with gr.Row():
        user_input = gr.Textbox(lines=2, label="Your Question", placeholder="e.g. What does Kanye say about mental health?")
        submit_btn = gr.Button("Ask")

    result = gr.Textbox(label="Answer", lines=8)

    with gr.Accordion("Sample Questions", open=False):
        gr.Markdown("""
        - What does Kanye think about creativity?
        - What role does God play in Kanye's life?
        - How does he talk about his bipolar disorder?
        - What does Kanye say about his childhood?
        - Why can’t humans breathe underwater?
        """)

    submit_btn.click(fn=kanye_rag_interface, inputs=user_input, outputs=result)

# Launch app
demo.launch()
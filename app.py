import gradio as gr
from PIL import Image
from kanye_base_code import retrieve_with_validation, build_prompt, client  # aus deiner Datei

# Hauptfunktion für die UI
def kanye_rag_interface(user_query):
    chunks, is_valid = retrieve_with_validation(user_query, k=5)
    if not is_valid:
        return "Diese Frage scheint nicht zum Kanye-Kontext zu passen."

    prompt = build_prompt(chunks, user_query)
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "Du bist ein hilfsbereiter Assistent mit Fokus auf Kultur, Psychologie & Kunst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# UI bauen
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Column():
        gr.Image("kanye.jpeg", show_label=False, container=False, elem_id="kanye-img")

        gr.Markdown("##  **KanyeRAG** – Kontextuelles Fragen an Kanye")
        gr.Markdown(" Ein semantisches RAG-System basierend auf Lyrics, Tweets, Biografie, Studien & Zitaten – powered by FAISS & LLaMA3")

    with gr.Row():
        user_input = gr.Textbox(lines=2, label="Deine Frage", placeholder="z.B. Was sagt Kanye über mentale Gesundheit?")
        submit_btn = gr.Button("Antwort generieren")

    result = gr.Textbox(label="Antwort", lines=8)

    with gr.Accordion("Beispiel-Fragen", open=False):
        gr.Markdown("""
        - Was denkt Kanye über Kreativität?
        - Welche Bedeutung hat Gott für Kanye?
        - Welche Rolle spielt seine Bipolarität?
        - Wie spricht Kanye über seine Kindheit?
        - Warum kann der Mensch nicht unter Wasser atmen?
        """)

    submit_btn.click(fn=kanye_rag_interface, inputs=user_input, outputs=result)

# App starten
demo.launch()

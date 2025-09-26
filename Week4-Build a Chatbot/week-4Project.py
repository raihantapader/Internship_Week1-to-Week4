!pip install --upgrade openai langchain langchain-openai python-dotenv
!pip install -U --force-reinstall langchain-openai

import gradio as gr
# from fastapi import FastAPI # Removed FastAPI as not needed
from datetime import datetime
# import uvicorn # Removed uvicorn as not needed
import os # Import os to access environment variables
import time # Import the time module


# Import your existing chat system
# from your_chat_system_file import ContextualChatSystem  # remove this line

# Copy the ContextualChatSystem class definition from cell k-y4EbrEd_39 here
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class ContextualChatSystem:
    def __init__(self, api_key: str):
        # Initialize GPT models (same for selection & generation)
        self.selector_model = ChatOpenAI(
            temperature=0.3,
            model="gpt-3.5-turbo",
            api_key=api_key
        )

        self.generator_model = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo",
            api_key=api_key
        )

        # Selector prompt
        self.selector_prompt = ChatPromptTemplate.from_template(
            """You are a context selection assistant.
Available contexts:
{contexts}

User question: {question}

Return ONLY the ID of the single most relevant context (e.g., 0, 1, 2).
If none are relevant, return "none"."""
        )

        # Generator prompt (improved)
        self.generator_prompt = ChatPromptTemplate.from_template(
            """You are an expert assistant answering strictly based on the provided context.

Relevant context:
{selected_context}

User question:
{question}

Instructions:
1. Answer the question clearly and concisely.
2. Add 1–2 extra helpful facts from the context (if available).
3. Always explain *which context* you used to answer.
4. If the context does not contain the answer, reply:
   "I don't have that information in my knowledge base."
5. Never invent facts.

Format your response as:
- ✅ Answer
- 📖 Context Used
- ℹ️ Extra Notes"""
        )

        # Chains
        self.context_selector = (
            {"contexts": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.selector_prompt
            | self.selector_model
            | StrOutputParser()
        )

        self.response_generator = (
            {"selected_context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.generator_prompt
            | self.generator_model
            | StrOutputParser()
        )

    def format_contexts(self, contexts):
        """Format contexts with IDs"""
        return "\n".join(f"ID: {i}\n{ctx}" for i, ctx in enumerate(contexts))

    def get_response(self, contexts, question):
        """Run context selection and response generation"""
        formatted_contexts = self.format_contexts(contexts)

        selected_context_id = self.context_selector.invoke({
            "contexts": formatted_contexts,
            "question": question
        }).strip()

        if selected_context_id.lower() == "none":
            selected_context = "No relevant context available"
        else:
            try:
                selected_context = contexts[int(selected_context_id)]
            except (ValueError, IndexError):
                selected_context = "No relevant context available"

        response = self.response_generator.invoke({
            "selected_context": selected_context,
            "question": question
        })

        return response


OPENAI_API_KEY = "your api key"  # keep your key
contexts = [
    "The capital of France is Paris. France is located in Western Europe and is known for its wine and cheese.",
    "The Python programming language was created by Guido van Rossum and first released in 1991. It emphasizes code readability.",
    "The human heart has four chambers: two atria and two ventricles. It pumps blood throughout the body."
]

chat_system = ContextualChatSystem(api_key=OPENAI_API_KEY)

# --- Gradio function ---
chat_history = []

def chat_gradio(user_message):
    global chat_history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Start timing AI response
    start_time = time.time()
    ai_response = chat_system.get_response(contexts, user_message)
    end_time = time.time()

    response_time = end_time - start_time

    # Add messages to history
    chat_history.append((f"[{timestamp}] User: {user_message}", f"[{timestamp}] AI: {ai_response} (Response time: {response_time:.2f}s)"))

    # Flatten chat for Gradio display
    display_history = []
    for u, a in chat_history:
        display_history.append(u)
        display_history.append(a)

    return "\n\n".join(display_history)

# --- Gradio interface ---
with gr.Blocks() as demo:
    gr.Markdown("## Contextual Chatbot")
    chatbot_output = gr.Textbox(label="Chat", placeholder="Type your message here...", lines=20, interactive=False)
    user_input = gr.Textbox(label="Your Message", placeholder="Type message here...", lines=1)
    send_btn = gr.Button("Send")

    send_btn.click(fn=chat_gradio, inputs=user_input, outputs=chatbot_output)
    user_input.submit(fn=chat_gradio, inputs=user_input, outputs=chatbot_output)

# Launch Gradio app directly
demo.launch(share=True)

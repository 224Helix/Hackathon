from flask import Flask, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

model = OllamaLLM(model='phi3')
prompt = ChatPromptTemplate.from_template(template)

# Initial hackathon context
initial_context = """
Hackathon Context:
Zeus has ordered the participants to find and stop the person causing mayhem on Olympus. The order is given in a code, which is a Caesarian cipher. 
Hera claims she is not the culprit and has noticed more deaths of married men, especially fishermen, in certain regions, hinting that Poseidon may be responsible.
Poseidon reveals he is involved but not behind everything. He mentions that a certain goddess has been getting on his nerves, causing the sea to be more chaotic, and provides a clue leading to Athena.

Bot's Role:
The bot's role is to answer questions from the participants about the hackathon without revealing the answers. The bot should provide helpful guidance and hints while ensuring the participants solve the challenges themselves.
"""

# Store contexts per user (in-memory, consider a more robust solution for production)
user_contexts = {}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_id = data.get('user_id')
    question = data.get('question')
    
    if not user_id or not question:
        return jsonify({'error': 'user_id and question are required'}), 400
    
    if user_id not in user_contexts:
        user_contexts[user_id] = initial_context
    
    context = user_contexts[user_id]
    
    try:
        generated_prompt = prompt.format(context=context, question=question)
        result = model.invoke(generated_prompt)
        answer = result['text'] if isinstance(result, dict) and 'text' in result else result
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    user_contexts[user_id] += f"\nUser: {question}\nBot: {answer}"
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

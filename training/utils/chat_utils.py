from IPython.display import clear_output

from utils.model_utils import generate

def hidden_input(prompt):
    user_input = input(prompt)

    clear_output()
    return user_input

def chat_system_only_tag(human, agent, prompt ,model, tokenizer):
    session = 0
    history = []
    while True:
        session += 1
        utter_num = 0
        session_history = ''
        while True:
            if utter_num % 2 == 0:
                tag = hidden_input(f"{human}\'s tag :")
                query = f"\n{human}: ({tag}) " 
                speaker = human
            else:
                tag = hidden_input(f"{agent}\'s tag :")
                query = f"\n{agent}: ({tag}) "
                speaker = agent
            if tag == 'end':
                prompt['text'] += session_history
                break

            
            input_ = tokenizer(prompt['text'] + session_history + query, return_tensors = 'pt').to(device)

            output = generate(model,tokenizer,
                                  input_,
                                  num_beams=1,
                                  num_return_sequences=1,
                                  max_new_tokens=100)
            
            utter = output.replace(prompt['text']+ session_history, '')
            for i in utter.split("\n"):
                if i != '':
                    utter_ = i
                    break
            
            session_history = session_history + utter_ + '\n'
            print(session_history)
            utter_num += 1

            
            
            
    
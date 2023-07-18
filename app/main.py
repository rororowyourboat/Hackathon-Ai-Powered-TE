import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space


st.set_page_config(
    page_title="Agent based token engineering",
    page_icon=":)",
    layout="wide",
)

with st.sidebar:
    st.title('Token Sales Agent')
    st.markdown('''
    ##  About
This is a prototype for an LLM interface that supports crypto projects in the design of a token sales proposal. Our agent helps you run scenarios based on a radCAD token supply model including token allocations, and vesting schedules. It has access to benchmarking data and supports you in running A/B tests, sensitivity analysis and more.
    ''')
    add_vertical_space(5)
    st.info('Made with ❤️ at the [Augment Hackathon](https://www.augmenthack.xyz/)'
            '\n\n'
            'by [Token Engineering Academy](https://tokenengineering.net/)'
            '\n\n'
            'Github [Hackathon-Ai-Powered-TE](https://github.com/rororowyourboat/Hackathon-Ai-Powered-TE)')

agent_number = st.selectbox('Select Agent', ['Agent 1', 'Agent 2'])

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm TokenChat, How may I help you?"]

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']


input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

## Applying the user input box
with input_container:
    user_input = get_text()


def generate_response(agent_number, prompt):
    agent_number = "0"
    response = ''
    # if agent_number == 'Agent 1':
    #     response = agent_demo_1(prompt)
    # elif agent_number == 'Agent 2':
    #     response = agent_demo_2(prompt)
    # else:
    #     st.error('Agent not found')
    return response


## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(agent_number, user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state['generated'][i], key=str(i))


import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from model import agent_demo_1, agent_demo_2

st.set_page_config(
    page_title="Token Sales Agent",
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
    st.image('img/tea.png', use_column_width=True)

agent_number = st.selectbox('Select Agent', ['Agent 1', 'Agent 2'])

# Store AI generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "I'm TokenChat, How may I help you?"}]

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def generate_response(agent_number, prompt):
    response = ''
    if agent_number == 'Agent 1':
        response = agent_demo_1(prompt)
    elif agent_number == 'Agent 2':
        response = agent_demo_2(prompt)
    else:
        st.error('Agent not found')
    return response


# Prompt for user input and save
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# If last message is not from assistant, we need to generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    # Call LLM
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(agent_number, prompt)
            st.write(response)

    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
import streamlit as st

st.title("Particle Trieur Training Portal")

st.write("Here you can launch PT training runs and monitor their progress.")

st.markdown(
    """
    ## Training
    
    Select **Training** from the sidebar to open the options to launch a training run.
    
    ## Monitoring
    
    Training runs are queued on the server. Select **Queue** from the sidebar to view the status of the queue.
    """
)

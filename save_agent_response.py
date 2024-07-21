from assistant_main import AgentResponse

agent_response = AgentResponse()
agent_response.save_message(
    response_message=("Sorry, I'm having trouble finding the right light."),
    file_name="light_not_found",
)

agent_response.play_message(file_name="light_not_found")

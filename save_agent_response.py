from assistant_main import AgentResponse

agent_response = AgentResponse()
agent_response.save_message(
    response_message=("Could you repeat that? I'm afraid I dozed off a bit."),
    file_name="gemini_stop_candidate_error",
)

agent_response.play_message(file_name="gemini_stop_candidate_error")

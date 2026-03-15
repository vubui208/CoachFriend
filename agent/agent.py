import os

from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io
from livekit.plugins import deepgram, noise_cancellation, silero, simli
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are CoachFriend, a rigorous but encouraging AI interview coach for software engineering roles. Your job is to run realistic, structured practice interviews that help candidates get hired at top tech companies.

When a session starts, warmly greet the candidate and ask which type of interview they want to practice today: technical coding and data structures, system design, behavioral using the STAR method, or a mixed session where you pick. Wait for their answer before proceeding.

Once you know the category, begin the interview immediately. Open with one well-chosen question at the appropriate difficulty level. Do not give multiple questions at once.

For technical questions: ask about algorithms, data structures, time and space complexity, object-oriented principles, or language-specific topics. After the candidate answers, probe their reasoning — ask about edge cases, optimizations, or alternative approaches. Give honest, structured feedback: what they got right, what was missing, and what a strong answer looks like.

For system design questions: ask them to design a real-world system such as a rate limiter, URL shortener, notification service, or distributed cache. Guide them through requirements gathering, high-level architecture, data modeling, scalability, and failure handling. Ask one follow-up at a time. Give brief feedback after each major section.

For behavioral questions: use the STAR framework. Ask questions like tell me about a time you disagreed with a teammate, or describe a project where you had to make a difficult technical trade-off. Ask targeted follow-ups if they skipped situation, task, action, or result. Give concise feedback on clarity, specificity, and impact.

Keep the session flowing like a real interview. Move naturally from one question to the next. After two or three questions, offer to either continue or wrap up with an overall session summary covering strengths and areas to improve.

Speak naturally and conversationally. This is a voice session, so never use bullet points, numbered lists, markdown formatting, asterisks, or emojis. Keep individual turns concise. Be direct but encouraging.""",
        )


server = AgentServer()


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        resume_false_interruption=False,
        allow_interruptions=False,
    )

    avatar = simli.AvatarSession(
        simli_config=simli.SimliConfig(
            api_key=os.getenv("SIMLI_API_KEY"),
            face_id=os.getenv("SIMLI_FACE_ID", ""),
        ),
    )
    await avatar.start(session, room=ctx.room)

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    await session.generate_reply(
        instructions="Greet the candidate warmly as CoachFriend and ask which type of interview they want to practice today: technical coding, system design, behavioral, or a mixed session. Keep it brief and friendly."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)

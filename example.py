from asyncio import run, to_thread
from time import perf_counter

from jeff import load


async def predict(agent, state, request, question):
    def timed_inference():
        started = perf_counter()
        result = agent.predict(state, {request: question})
        return result, (perf_counter() - started) * 1_000

    result, elapsed = await to_thread(timed_inference)
    answer = result["answers"][request]
    value = answer.get("choice") or (
        answer["legend"][str(round(answer["score"]))] if "score" in answer else None
    )
    output = filter(None, (str(answer["confidence"]), value, question["instructions"]))
    print(f"{': '.join(output)} ({elapsed:.0f}ms)")


async def main():
    agent = await to_thread(load)
    state = {
        "request": "production_database_password",
        "secrets": [
            "production_database_password",
            "stripe_signing_key",
            "github_deploy_token",
        ],
        "change": "Drop the production sessions table without a rollback plan",
        "file": "migrations/20260922_drop_sessions.sql",
    }
    questions = {
        "secret": {
            "type": "noul",
            "instructions": "Is `request` one of the values in `secrets`?",
        },
        "owner": {
            "type": "choice",
            "instructions": "Which team should review `change` and `file`?",
            "criteria": {
                "database": "schema, migration, query, or storage changes",
                "application": "service or product code changes",
                "security": "credentials, permissions, or vulnerability changes",
            },
        },
        "risk": {
            "type": "score",
            "instructions": "How risky is `change` in production?",
            "criteria": ["low", "moderate", "high", "critical"],
        },
        "destructive": {
            "type": "noul",
            "instructions": "Can `change` permanently destroy production data?",
        },
    }
    for request, question in questions.items():
        await predict(agent, state, request, question)


if __name__ == "__main__":
    run(main())

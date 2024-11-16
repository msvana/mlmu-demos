import openai
from pydantic import BaseModel, Field


client = openai.OpenAI()

prompt = """\
You are an expert on evaluating brainstorming ideas. You will be given a problem to solve,
and an idea proposed to solve that problem. Your task is to evaluate the idea according to
the following criteria:

- Originality: Is the idea original and creative?
- Feasibility: Is the idea feasible to implement?
- Usefulness: Is the idea useful in solving the problem?

Provide a detailed explanation of your evaluation of the idea and then give a score
from 1 to 10 for each of the three criteria. Be very critical and nitpicky.

<problem>{problem}</problem>
<idea>{idea}</idea>
"""


class IdeaEvaluation(BaseModel):
    reasoning: str = Field(description="Thinking process leading to the evaluation")
    originality: int = Field(description="Score from 1 to 10")
    feasibility: int = Field(description="Score from 1 to 10")
    usefulness: int = Field(description="Score from 1 to 10")


def evaluate_idea(problem: str, idea: str) -> IdeaEvaluation:
    prompt_filled = prompt.format(problem=problem, idea=idea)
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt_filled}],
        response_format=IdeaEvaluation,
        temperature=0,
    )

    if not completion.choices[0].message.parsed:
        raise ValueError("No evaluation was returned")

    evaluation = completion.choices[0].message.parsed
    return evaluation


problem_to_solve = "How to improve university education in the Czech Republic?"

prompt = f"""\
You are currently attending a brainstorming session. You are proposing 
ideas to solve the following problem:

{problem_to_solve}

It is your turn to propose an idea. Write a short paragraph describing
your idea. The idea should be original, feasible, and useful.
"""


class Idea(BaseModel):
    reasoning: str = Field(description="Thinking process leading to the idea")
    idea: str = Field(description="The idea itself")


completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    response_format=Idea,
    temperature=0.75,
)

idea = completion.choices[0].message.parsed

assert idea is not None

print(idea.idea)

evaluation = evaluate_idea(problem_to_solve, idea.idea)

print(evaluation)

assert evaluation.originality >= 8
assert evaluation.feasibility >= 8
assert evaluation.usefulness >= 8

import openai
from pydantic import BaseModel, Field


client = openai.OpenAI()

question_prompt = """\
You are an expert on evaluating how well a person performed a certain task.
You will be given the task description, and the person's response to the task.
Then you will anwer a yes/no question about the person's response. 

For example:

Task description: Write a title for a webpage that describes the current state of the page.
Person's response: "Login and registration page - Forms empty, no errors, user logged out"
Does the response mention that the user is logged out? "yes"

<task-description>
{task}
</task-description>

<person-response>
{response}
</person-response>

<question>
{question}
</question>
"""


class Answer(BaseModel):
    reasoning: str = Field(description="Thinking process leading to the answer")
    answer: str = Field(description="The answer itself (lowercase yes or no)")


def ask_question(task: str, response: str, question: str) -> bool:
    prompt_filled = question_prompt.format(task=task, response=response, question=question)

    client = openai.OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_filled}],
        response_format=Answer,
        temperature=0,
    )

    if not completion.choices[0].message.parsed:
        raise ValueError("Failed to parse the completion")

    answer = completion.choices[0].message.parsed

    if answer.answer not in ["yes", "no"]:
        raise ValueError(f"Unexpected answer: {answer.answer}")

    return answer.answer == "yes"


with open("data/hackernews.html") as fd_html:
    html = fd_html.read()



prompt = f"""\
You are an expert on webpage semantics. Your task is to analyze the following piece of HTML
and write a comprehensive description of the webpage. The description should contain information
about the most important elemements of the page and a list of actions a user can perform on the page.
Don't be too technical. Instead of describing the page to a developer, describe it to a regular user.

{html}
"""

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=1024,
    temperature=0,
)

description = completion.choices[0].message.content

assert description is not None
assert ask_question(prompt, description, "Does the description that there is a login form?")
assert ask_question(prompt, description, "Does the description mention that there is a registration form?")
assert ask_question(prompt, description, "Does the description mention that the user can reset the password?")
assert not ask_question(prompt, description, "Does the description mention that the webpage has a favicon?")

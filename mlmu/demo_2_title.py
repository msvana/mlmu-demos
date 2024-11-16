import numpy as np
import openai
from pydantic import BaseModel, Field


with open("data/hackernews.html") as fd_html:
    html = fd_html.read()

title_expected = "Login and registration page - Forms empty, no errors, user logged out"


def cosine_similarity(emb_a: openai.types.Embedding, emb_b: openai.types.Embedding) -> float:
    emb_a_arr = np.array(emb_a.embedding)
    emb_b_arr = np.array(emb_b.embedding)
    dot_product = np.dot(emb_a_arr, emb_b_arr)

    norm_a = np.linalg.norm(emb_a_arr)
    norm_b = np.linalg.norm(emb_b_arr)

    sim = dot_product / (norm_a * norm_b)
    return sim


class Title(BaseModel):
    reasoning: str = Field(description="Thinking process leading to the suggested title")
    title: str = Field(description="The title itself")


client = openai.OpenAI()

prompt = f"""\
You are an expert on webpage semantics.
Analyze the following piece of HTML and return a JSON object containing the title 
you would suggest for this webpage. The title shouldn't simply match the contents
of the <title> tag. I want you to suggest a title that describes what you actually see.
Most importantly, it should contain description of the current state dynamic properties,
for example:

- user status (logged in or logged out)
- form field status (password field filled, empty)
- status of lists of objects (list has items, list is empty)
- status of a shopping cart (empty, with items)
- visible error messages (password incorrect, no errors)
- and much more.

Here are a few examples of a good title:
<title>List of todos - Empty, user logged in</title>
<title>Shopping cart - With items</title>
<title>Login page - Username filled, User doesn't exist error</title>

Here is the HTML for you to analyze:

{html}
"""

completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    response_format=Title,
    temperature=0,
    seed=100
)

title_generated = completion.choices[0].message.parsed
assert title_generated is not None
print(title_generated.title)

embeddings = client.embeddings.create(
    input=[title_expected, title_generated.title], model="text-embedding-3-small"
)
similarity = cosine_similarity(embeddings.data[0], embeddings.data[1])

try:
    assert similarity >= 0.85
except AssertionError as e:
    print(f"Similarity value {similarity:.3f} less than 0.85")
    raise e

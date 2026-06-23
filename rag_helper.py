from google.genai import types
from google.genai.errors import ClientError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception

INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course='llm-zoomcamp',
        model='gemini-2.0-flash'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        boost_dict = {'question': 3.0, 'section': 0.5}
        filter_dict = {'course': self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, question, search_results):
        context = self.build_context(search_results)
        prompt = self.prompt_template.format(
            question=question,
            context=context
        )
        return prompt.strip()

    def llm(self, user_prompt):
        @retry(
            retry=retry_if_exception(lambda e: isinstance(e, ClientError) and e.code == 429),
            wait=wait_exponential(multiplier=1, min=20, max=120),
            stop=stop_after_attempt(5),
            reraise=True
        )
        def _call():
            response = self.llm_client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.instructions
                )
            )
            return response.text

        return _call()

    def rag(self, question):
        search_results = self.search(question)
        user_prompt = self.build_prompt(question, search_results)
        return self.llm(user_prompt)
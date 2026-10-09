
from pydantic import BaseModel, Field, EmailStr, ValidationError
import os
import json

from dotenv import load_dotenv
from openai import OpenAI



class Education(BaseModel):
    degree: str | None = None
    institution: str | None = None
    graduation_year: int | None = None

class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None

class Resume(BaseModel):
    name: str | None = None
    phone: str | None = None
    email: EmailStr | None = None
    location: str | None = None
    skills: list[str] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    summary: str | None = None



load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def build_prompt(resume_text: str, errors: str = "") -> str:
    schema = Resume.model_json_schema()

    return f"""
You are an expert resume information extraction system.

Extract structured information from the provided resume.

Rules:
1. Extract only information supported by the resume.
2. Never invent missing information.
3. Use null for missing optional fields.
4. Use empty lists for missing skills, education,
   and experience.
5. graduation_year must be an integer or null.
6. Return a single JSON object matching the schema.
7. Correct any previous validation errors.

Required JSON schema:
{json.dumps(schema, indent=2)}

Resume text:
<resume>
{resume_text}
</resume>

Previous validation errors:
{errors}

Return only the JSON object.
"""

def extract_resume(resume_text: str, max_retries: int = 3,) -> Resume:
    errors = ""
    for attempt in range(max_retries + 1):
        prompt = build_prompt(resume_text, errors)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content":  "Extract resume information into JSON. "
                        "Never invent missing information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            response_format = {
                "type": "json_object",
            },
            stream=True
        )

        # raw_output = response.choices[0].message.content
        raw_output = ""

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                raw_output += content

        print()
        try:

            if not raw_output:
                raise ValueError("No output from model")


            data = json.loads(raw_output)
            resume = Resume.model_validate(data)

            print(f"Validation Successful on attempt {attempt + 1}")
            return resume

        except (ValidationError, ValueError) as e:
            errors = str(e)
            print(f"Validation Error on attempt {attempt + 1}: {errors}")

            if (attempt == max_retries):
                raise ValueError(f"Failed to extract valid resume after {max_retries} attempts. Last error: {errors}")

    raise RuntimeError("Unexpected error in extract_resume function")


print("Paste your resume below. Type END on a separate line when finished.")

lines = []

while True:
    line = input()
    if line.strip() == "END":
        break
    lines.append(line)

resume_text = "\n".join(lines).strip()

try:
    result = extract_resume(resume_text)
    print("Extracted Resume:")
    print(result.model_dump_json(indent=2))
except ValueError as e:
    print(f"Error: {e}")
import os
import json
import base64

from openai import OpenAI


class QwenVLM:

    def __init__(self):

        self.client = OpenAI(
            api_key=os.environ[
                "DASHSCOPE_API_KEY"
            ],
            base_url=(
                "https://dashscope.aliyuncs.com/"
                "compatible-mode/v1"
            ),
        )

        self.model = "qwen3.8-flash"


    def _encode_image(self, image_path):

        with open(
            image_path,
            "rb"
        ) as f:

            return base64.b64encode(
                f.read()
            ).decode("utf-8")


    def decide(
        self,
        image_path,
        instruction,
        history
    ):

        image_base64 = self._encode_image(
            image_path
        )

        history_text = json.dumps(
            history,
            ensure_ascii=False,
            indent=2
        )

        prompt = f"""
    You are the high-level decision-making agent for a robot arm.

    Your job is to decide ONE next action.

    ==================================================
    USER GOAL
    ==================================================

    {instruction}


    ==================================================
    AVAILABLE TOOLS
    ==================================================

    1. pick(target)

    Pick up an object.

    Example:

    {{
        "function": "pick",
        "target": "red cube",
        "reason": "The cube must be picked up first."
    }}


    2. place(target)

    Move the currently held object into a target region
    and release it.

    Example:

    {{
        "function": "place",
        "target": "green goal",
        "reason": "The cube is already held and must be moved to the goal."
    }}


    3. finish()

    Call this ONLY when the user's complete goal
    has already been achieved.

    Example:

    {{
        "function": "finish",
        "target": null,
        "reason": "The requested task has been completed."
    }}


    ==================================================
    ACTION HISTORY
    ==================================================

    {history_text}


    ==================================================
    CURRENT OBSERVATION
    ==================================================

    A current camera image is attached.


    ==================================================
    IMPORTANT RULES
    ==================================================

    - Decide exactly ONE next action.
    - Use the action history to understand what has already happened.
    - Do not repeat a successful action unnecessarily.
    - If pick succeeded, do not pick the same object again.
    - If place succeeded and this satisfies the user's goal, call finish.
    - A successful tool result is reliable.
    - Do not invent tools.
    - Return ONLY valid JSON.
    - Do not include markdown.
    - Do not include ```json.
    - Do not explain anything outside the JSON.

    Return:

    {{
        "function": "pick | place | finish",
        "target": "target name or null",
        "reason": "short reason"
    }}
    """

        response = self.client.chat.completions.create(
            model=self.model,

            messages=[
                {
                    "role": "user",

                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },

                        {
                            "type": "image_url",

                            "image_url": {
                                "url":
                                    "data:image/png;base64,"
                                    + image_base64
                            },
                        },
                    ],
                }
            ],
        )

        text = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        print()
        print("[Qwen raw response]")
        print(text)

        if text.startswith("```"):

            text = text.replace(
                "```json",
                ""
            )

            text = text.replace(
                "```",
                ""
            )

            text = text.strip()

        return json.loads(text)
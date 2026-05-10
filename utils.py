import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer
from tabulate import tabulate

# =========================================================
# Load Environment Variables
# =========================================================

load_dotenv()

# =========================================================
# Model + Tokenizer
# =========================================================

base_model_id = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# =========================================================
# System Prompt
# =========================================================

SYSTEM_PROMPT = """
You are playing Wordle, a word-guessing game.

### Game Rules:
- You have 6 tries to guess a secret 5-letter word.
- Each guess must be a valid 5-letter English word.
- After each guess, you will receive feedback.

### Feedback Format:
✓ : Correct letter in correct position
- : Correct letter in wrong position
x : Letter not in the word

### Response Format:
First think step-by-step inside <think></think> tags.
Then return your guess inside:
<guess> guessed-word </guess>
"""

# =========================================================
# Groq Client
# =========================================================

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

# =========================================================
# Generate Stream
# =========================================================

def generate_stream(
    prompt: str,
    adapter_id: str = "llama-3.3-70b-versatile",
    temperature: float = 0.7,
    max_tokens: int = 1024,
):

    response = client.chat.completions.create(
        model=adapter_id,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n<think>"
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    completion = ""

    for chunk in response:

        content = chunk.choices[0].delta.content

        if content is not None:
            print(content, end="", flush=True)
            completion += content

    print()

    return completion

# =========================================================
# Generate Multiple Responses
# =========================================================

def generate(
    messages: List[dict],
    adapter_id: str = "llama-3.3-70b-versatile",
    num_guesses: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 1024,
):

    outputs = []

    for i in range(num_guesses):

        try:

            completion = client.chat.completions.create(
                model=adapter_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            outputs.append(
                completion.choices[0].message.content
            )

        except Exception as e:

            print(f"Error on guess {i+1}: {e}")

    return outputs
# =========================================================
# Feedback Enum
# =========================================================

class LetterFeedback(Enum):
    CORRECT = "✓"
    WRONG_POS = "-"
    WRONG_LETTER = "x"

# =========================================================
# Wordle Feedback Logic
# =========================================================

def get_feedback(
    guess: str,
    secret_word: str
) -> List[LetterFeedback]:

    valid_letters = set(secret_word)

    feedback = []

    for letter, secret_letter in zip(guess, secret_word):

        if letter == secret_letter:
            feedback.append(LetterFeedback.CORRECT)

        elif letter in valid_letters:
            feedback.append(LetterFeedback.WRONG_POS)

        else:
            feedback.append(LetterFeedback.WRONG_LETTER)

    return feedback

# =========================================================
# Guess DataClass
# =========================================================

@dataclass
class GuessWithFeedback:

    guess: str
    feedback: List[LetterFeedback]

    def __repr__(self) -> str:

        feedback_str = " ".join(
            f"{letter}({fb.value})"
            for letter, fb in zip(self.guess, self.feedback)
        )

        return f"{self.guess} → Feedback: {feedback_str}"

    @staticmethod
    def from_secret(
        guess: str,
        secret: str
    ) -> "GuessWithFeedback":

        return GuessWithFeedback(
            guess,
            get_feedback(guess, secret)
        )

# =========================================================
# User Prompt
# =========================================================

def render_user_prompt(
    past_guesses: List[GuessWithFeedback]
):

    prompt = "Make a new 5-letter word guess."

    if past_guesses:

        prompt += "\n\nHere is previous feedback:"

        for i, guess in enumerate(past_guesses):

            prompt += f"\nGuess {i+1}: {guess}"

    return prompt

# =========================================================
# Build Messages
# =========================================================

def get_messages(
    past_guesses: List[GuessWithFeedback]
):

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": render_user_prompt(past_guesses)
        }
    ]

# =========================================================
# Render Prompt
# =========================================================

def render_prompt(
    past_guesses: List[GuessWithFeedback]
):

    messages = get_messages(past_guesses)

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )

# =========================================================
# Extract Guess
# =========================================================

def extract_guess(completion: str) -> str:

    match = re.search(
        r"<guess>\s*([\s\S]*?)\s*</guess>",
        completion,
        re.DOTALL
    )

    if not match:
        return ""

    return match.group(1).strip().upper()

# =========================================================
# Next Turn
# =========================================================

def next_turn(
    past_guesses: List[GuessWithFeedback],
    secret_word: str,
    adapter_id="llama-3.3-70b-versatile"
):

    prompt = render_user_prompt(past_guesses)

    completion = generate_stream(
        prompt,
        adapter_id=adapter_id
    )

    guess = extract_guess(completion)

    feedback = get_feedback(
        guess,
        secret_word
    )

    past_guesses.append(
        GuessWithFeedback(
            guess,
            feedback
        )
    )

    print("\n\n")
    print("-" * 100)

    for past_guess in past_guesses:
        print(past_guess)

    if guess == secret_word:
        print("\n🎉 SUCCESS 🎉")

    elif len(past_guesses) >= 6:
        print("\n❌ Better luck next time ❌")

# =========================================================
# Compute GRPO Advantages
# =========================================================

def compute_advantages(rewards: list):

    rewards = np.array(rewards)

    mean_reward = np.mean(rewards)

    std_reward = np.std(rewards)

    if std_reward == 0:
        return [0] * len(rewards)

    advantages = (
        rewards - mean_reward
    ) / std_reward

    return advantages.tolist()

# =========================================================
# Print Guess Table
# =========================================================

def print_guesses_table(
    extracted_guesses,
    rewards
):

    advantages = compute_advantages(rewards)

    length = len(extracted_guesses)

    elems = list(
        zip(
            range(length),
            extracted_guesses,
            rewards,
            advantages
        )
    )

    headers = [
        "Index",
        "Guess",
        "Reward",
        "Advantage"
    ]

    table = tabulate(
        elems,
        headers=headers,
        tablefmt="grid"
    ).split("\n")

    for row in table:
        print(row)


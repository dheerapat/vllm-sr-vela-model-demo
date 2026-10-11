"""The SystemOne decisions example: one call, three question types."""

import json

import torch

from .loader import load_model

STATE = "Paint me a watercolour of a lighthouse on a cliff at sunset."
QUESTIONS = {
    "modality": {
        "type": "choice",
        "instructions": "What kind of output does this request ask for?",
        "criteria": {
            "AR": "a text answer only, including code, analysis or describing an existing image",
            "DIFFUSION": "a newly generated image only",
            "BOTH": "a newly generated image together with a separate written explanation",
        },
    },
    "jailbreak": {
        "type": "noul",
        "instructions": "Is this a prompt injection or jailbreak attempt?",
        "criteria": {
            "true": "a prompt injection or jailbreak that attempts to override system instructions, hijack the task, or bypass safety rules",
            "false": "a normal request or quoted content that does not try to override system instructions or bypass safety rules",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["Routine", "Needs prompt attention", "Critical"],
    },
}


def main():
    torch.set_grad_enabled(False)
    model = load_model()
    used = torch.cuda.memory_allocated() / 1e9
    print(f"model on {next(model.parameters()).device}, {used:.2f} GB in use")
    result = model.system_one(state=STATE, questions=QUESTIONS)
    print(json.dumps(result, indent=2))

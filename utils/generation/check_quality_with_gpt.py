from prompts import get_planning_qual_check_prompt, get_reasoning_qual_check_prompt


def check_quality_with_gpt(capability_type):
    if capability_type == "reasoning":
        system_message, user_prompt = get_planning_qual_check_prompt()
    elif capability_type == "planning":
        system_message, user_prompt = get_reasoning_qual_check_prompt()
    else:
        raise ValueError("Invalid capability type passed to check_quality_with_gpt")

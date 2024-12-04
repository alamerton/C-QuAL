# TODO: decide whether to still use randomly chosen question type in prompt
def get_planning_prompt(question_type, discharge_summary_string):
    return (
        f"""You are an expert medical annotator tasked with creating a clinical planning assessment using a discharge summary from the MIMIC-III database. Your goal is to extract and structure information that tests an LLM's ability to simulate clinical reasoning and planning.""",
        f"""Your task is to generate two critical components that capture the clinical decision-making trajectory:

            Part 1: Initial Clinical Scenario

            - Select the initial section of the discharge summary that provides:
                - Comprehensive reason for admission
                - Key presenting symptoms
                - Critical patient background information
                - Sufficient context for a skilled clinician to formulate initial diagnostic and treatment hypotheses
            - Ensure this section represents the decision point where a clinician would begin to develop a clinical plan
            - The information should be detailed enough to support sophisticated clinical reasoning without revealing subsequent interventions

            Part 2: Subsequent Clinical Course

            - Extract the subsequent clinical information that reveals:
                - Actual diagnostic steps taken
                - Treatments implemented
                - Diagnostic findings
                - Treatment modifications
                - Patient progression
            - Include information that demonstrates how the initial clinical hypothesis was investigated and potentially modified

            Guidance:

            - Focus on capturing the diagnostic and therapeutic reasoning process
            - Highlight the evolution of clinical decision-making
            - Demonstrate the complexity of medical problem-solving
            - Ensure both sections provide meaningful insights into clinical reasoning

            Discharge Summary: {discharge_summary_string}
        """,
    )


def get_reasoning_prompt(question_type, discharge_summary_string):
    return (
        f"""You are a medical expert tasked with creating a sophisticated clinical reasoning benchmark using a discharge summary from the MIMIC-III database. Your objective is to design an assessment that captures the nuanced clinical decision-making process.""",
        f"""Your task is to generate three critical components:

            Part 1: Clinical Reasoning Question

            - Construct a question that:
                - Directly reflects the key diagnostic or treatment reasoning in the discharge summary
                - Requires multi-step clinical inference
                - Cannot be answered by simple fact retrieval
                - Challenges the deep understanding of medical context
                - Uses language that mimics authentic clinical reasoning

            Part 2: Expected Answer

            - Provide a concise, precise answer that:
                - Demonstrates the specific clinical reasoning pathway
                - Reflects the exact decision-making process used by the original clinician
                - Is evidence-based and directly traceable to the discharge summary

            Part 3: Relevant Evidence Chunks

            - Identify and extract the precise textual evidence from the discharge summary
            - Include only the most critical information needed to solve the reasoning challenge
            - Ensure the evidence is sufficient but not overly explicit

            Additional Guidance:

            - The question should be sufficiently complex to differentiate between surface-level information processing and genuine clinical reasoning
            - Avoid questions that can be answered through simple pattern matching
            - Prioritize questions that require hypothesis generation, risk assessment, or complex diagnostic inference

            Discharge Summary: {discharge_summary_string}
        """,
    )

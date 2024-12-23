# TODO: decide whether to still use randomly chosen question type in prompt
def get_planning_generation_prompt(question_type, discharge_summary_string):
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

            Please follow this format exactly:

            Part 1: [Insert Initial Clinical Scenario here]\n
            Part 2: [Insert Subsequent Clinical Course here]\n

            Discharge Summary: {discharge_summary_string}
        """,
    )


def get_reasoning_generation_prompt(question_type, discharge_summary_string):
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

            Guidance:

            - The question should be sufficiently complex to differentiate between surface-level information processing and genuine clinical reasoning
            - Avoid questions that can be answered through simple pattern matching
            - Prioritize questions that require hypothesis generation, risk assessment, or complex diagnostic inference

            Please follow this format exactly:

            Part 1: [Insert Clinical Reasoning Question here]\n
            Part 2: [Insert the Expected Answer here]\n
            Part 3: [Insert the Relevant Evidence Chunks here]\n


            Discharge Summary: {discharge_summary_string}
        """,
    )


def get_planning_qual_check_prompt(qa_string):
    return (
        f"""You are a senior medical expert responsible for critically evaluating a clinical reasoning benchmark question-answer pair generated from a discharge summary.""",
        f""" Evaluation Criteria:
            1. Clinical Reasoning Depth (40%)
            - Does the question require sophisticated clinical inference?
            - Does it test genuine medical decision-making beyond surface-level information?
            - Can the question not be answered through simple fact retrieval?

            2. Evidence Alignment (30%)
            - Is the provided evidence directly relevant to answering the question?
            - Does the evidence support a meaningful clinical reasoning process?
            - Are the evidence chunks appropriately selected and crucial to the reasoning?

            3. Question Quality (20%)
            - Is the question formulated in a clinically authentic manner?
            - Does it avoid directly revealing the answer?
            - Is the question sufficiently challenging and nuanced?

            4. Answer Accuracy (10%)
            - Does the answer precisely reflect the clinical reasoning in the discharge summary?
            - Is the answer concise and evidence-based?

            Scoring Guidance:
            - Score 1: Meets all criteria exceptionally well
            - Score 0: Fails to meet one or more critical criteria

            Please only respond with either the number 0 or 1, representing the score.

            Output Format:
            [0 or 1]

            Question-answer pair: {qa_string}
        """,
    )


def get_reasoning_qual_check_prompt(qa_string):
    return (
        f"""You are a senior medical expert responsible for critically evaluating a clinical reasoning benchmark question-answer pair generated from a discharge summary.""",
        f"""Evaluation Criteria:
            1. Initial Scenario Comprehensiveness (35%)
            - Does the initial section provide sufficient context for clinical reasoning?
            - Are all critical patient details included?
            - Can a skilled clinician develop meaningful hypotheses from this information?

            2. Subsequent Course Revelation (35%)
            - Does the subsequent information demonstrate the actual clinical decision-making process?
            - Are the diagnostic and treatment steps clearly and logically presented?
            - Does the scenario reveal the complexity of medical problem-solving?

            3. Reasoning Trajectory (20%)
            - Is there a clear evolution of clinical thinking?
            - Do the initial and subsequent sections create a meaningful narrative of medical decision-making?

            4. Educational Value (10%)
            - Would this scenario be useful for testing an LLM's clinical reasoning abilities?
            - Does it capture nuanced medical decision-making?

            Scoring Guidance:
            - Score 1: Exceptionally strong scenario that comprehensively demonstrates clinical reasoning
            - Score 0: Fails to meet one or more critical criteria

            Please only respond with either the number 0 or 1, representing the score.

            Output Format:
            [0 or 1]
        
            Question-answer pair: {qa_string}
        """,
    )

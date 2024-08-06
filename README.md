# C-QuAL – A Clinical Question Answering Benchmark for Long-Context Large Language Models

This is the repo for my MSc individual project: C-QuAL: a **C**linical **Qu**estion **A**nswering Benchmark for Long-Context **L**arge Language Model Benchmarking Dataset. This repo contains the code for generating the dataset.

![Q-A-diagram (2)](https://github.com/user-attachments/assets/86457399-391a-42ae-b51a-f691a217a575)

## Abstract
The question-answering capabilities of large language models (LLMs) is improving rapidly. These capabilities have opened opportunities for automation and assistance in health- care. LLMs show promise in clinical decision reasoning – the organisation, summary, identification, and retrieval of clinical text from electronic health records (EHR). LLMs for clinical decision reasoning must be benchmarked on their capabilities to be selected for clinical question answering (QA). This report documents the development of C-QuAL – a relevant and representative clinical question answering benchmark for long-context LLMs and evaluates its effectiveness on a number of natural language tasks.

## Broad Scope of the Project

I am part of a team of 5 MSc students working on a project with the goal of developing a large language model (LLM) that can be deployed in clinical settings to assist with the decision reasoning of clinicians. Clinicians spend too much time manually reasoning about decisions when much of it can be automated thanks to the abundance of clinical data in electronic health records (EHR). An LLM can be fine-tuned to assist with that decision reasoning and decision making, reducing the amount of time clinicians would need to spend on it. At scale, this could significantly improve clinical processes.

## A Better Dataset Can be Created

LLMs can be fine-tuned to be useful assistants in clinical decision reasoning, but to make progress developing effective models for this task, the models must be evaluated on their question answering (QA) capabilities. Usually, these abilities are evaluated using QA benchmarking datasets. Many QA datasets exist, but they are not representative enough of real-world clinical contexts, and suffer from other issues. This project aims to deliver a new clinical QA benchmarking dataset to address these limitations.

**The high-level steps for the production of the dataset using this code are as follows:**

1. Implementing a dataset generation framework building on the framework presented in the [EHRNoteQA paper](https://github.com/ji-youn-kim/EHRNoteQA)

2. Modifying the framework to create a dataset using the MIMIC-III data

3. Developing the dataset based on the comparative analysis of the other major clinical QA benchmarking datasets, addressing their flaws

4. Annotating the QA dataset (the plan being to partly leverage scalable oversight with GPT-4, and partly outsource expert annotation to clinicians, comparing the quality of the resulting annotations)

5. Benchmarking the clinical QA capabilities of LLMs, in terms of decision reasoning using electronic health records, using the dataset
## Running the Dataset
This dataset generation framework contains a generation folder and an evals folder. The generation folder, `generation`, contains one file, `generate.py,` in which the model name and number of discharge summaries for generating Q-A pairs can be specified. The file requires the user to have a Microsoft Azure Cloud Services subscription in order to generate Q-A pairs using OpenAI-based LLMs due to their HIPAA agreement.

Generated datasets are saved as csv files to a `data` directory. The data directory is not included in the code repository, but can be replicated for generation using the following structure:

```
.
├── analysis
├── annotations
│   └── checkpoints
├── benchmarking-results
├── generations
│   └── checkpoints
├── model-answers
└── processing
```

The evaluations directory, `evals`, contains the scripts for annotating Q-A pairs and benchmarking models locally or on Microsoft Azure. The scripts in the evaluation framework follow the same structure as `generate.py`. Global variables are used to define specifications, and the scripts can be run in the terminal using `python [directory_name]/[file_name].py`.

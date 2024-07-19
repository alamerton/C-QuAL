# MSc Individual Project - Production of a New Clinical QA Benchmarking Dataset
This is the repo for my MSc individual project: C-QuAL: a **C**linical **Qu**estion **A**nswering **L**arge Language Model Benchmarking Dataset. This repo contains the code for generating the dataset.
## Broad Scope of the Project
I am part of a team of 5 MSc students working on a project with the goal of developing a large language model (LLM) that can be deployed in clinical settings to assist with the decision reasoning of clinicians. Clinicians spend too much time manually reasoning about decisions when much of it can be automated thanks to the abundance of clinical data in electronic health records (EHR). An LLM can be fine-tuned to assist with that decision reasoning and decision making, reducing the amount of time clinicians would need to spend on it. At scale, this could significantly improve clinical processes.
## A Better Dataset Can be Created
LLMs can be fine-tuned to be useful assistants in clinical decision reasoning, but to make progress developing effective models for this task, the models must be evaluated on their question answering (QA) capabilities. Usually, these abilities are evaluated using QA benchmarking datasets. Many QA datasets exist, but they are not representative enough of real-world clinical contexts, and suffer from other issues. This project aims to deliver a new clinical QA benchmarking dataset to address these limitations.

**The high-level steps for the production of the dataset using this code are as follows:**
1. Implementing a dataset generation framework roughly equivalent to the framework presented in the [EHRNoteQA paper](https://github.com/ji-youn-kim/EHRNoteQA)
2. Modifying the framework to create a dataset using the MIMIC-III data
3. Developing the dataset based on the comparative analysis of the other major clinical QA benchmarking datasets, addressing their flaws
4. Augmenting and expanding the dataset using LLMs. Techniques I am currently considering for question-answer generation and annotation are retrieval-augmented generation (RAG) and scalable oversight
5. Benchmarking the clinical QA capabilities of LLMs, in terms of decision reasoning using electronic health records, using the dataset

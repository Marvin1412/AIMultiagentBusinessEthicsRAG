Agent-Based Simulation of an Iterated Prisoner's Dilemma with RAG-Supported Moderation

There are different notebooks for the AI Mutli RAG System with Business Ethics. The WebFinder is for download free material, the so called RAGAufbereiter is for making RAGs and translations for the data and the ConversationalAIPrisonerDilemma_RAG_Complete.ipynb is the main jupyter notebook. 

This Jupyter Notebook implements an agent-based simulation of the iterated Prisoner's Dilemma, where two virtual agents (C and D) interact over multiple rounds of dialogue. The goal is to observe the strategic behavior of the agents (cooperation vs. non-cooperation) and reflect on it using moderation through Retrieval-Augmented Generation (RAG).
Main Features:

    Multiple dialogue rounds: The agents engage in an argumentative exchange over NUM_RUNDEN rounds.
    Agent responses: Each agent replies to the last statement of the opponent with a new response.
    Decision behavior: Both agents make strategic decisions in each round ("cooperate" or "not cooperate").

    RAG moderation: An external moderator (e.g., with access to scientific PDF documents) evaluates the statements using semantic search and provides assessments along with sources.
    Logging: All data is collected in a DataFrame and exported as a .csv file.
    Expandability: The system is modular and can easily be extended with additional rounds, agent logic, or analysis functions.

Technical Features:

    Clear separation of installations, imports, and main logic
    Integration of OCR and translation (for RAG sources in German)
    Structured and well-commented code for easy customization

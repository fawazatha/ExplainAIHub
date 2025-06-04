# ExplainAIHub: Advanced AI Explanations, Simplified

> **Advanced AI Explanations, Simplified**  
>
> ExplainAIHub is a Python framework that unifies and streamlines explainability for machine learning models. By integrating multiple XAI tools under a common schema and leveraging LLM-powered summarization, it delivers clear, structured insights—both as static reports and conversational dialogues.

---

## The Problem We Solve

Understanding *why* a machine learning model makes a certain prediction is critical for trust, debugging, and responsible AI. While many XAI (Explainable AI) tools like SHAP and LIME exist, they:
1.  Produce outputs in diverse, often complex formats.
2.  Lack a common language, making it hard to synthesize insights if multiple tools are used.
3.  Require significant effort to translate their numerical outputs into human-understandable narratives.

ExplainAIHub addresses these challenges by providing a structured, multi-layered approach to AI explainability.

## Key Features

*   **Multi-Tool Integration:** Designed to run various popular XAI tools (starting with SHAP, with LIME and others planned).
*   **Standardized Explainable Insight Schema (SEIS):** A core innovation, SEIS translates raw, tool-specific XAI outputs into a common, rich, and structured format. This enables consistent processing and interpretation.
*   **LLM-Powered Summarization:** Leverages Large Language Models (LLMs) with carefully crafted prompts to generate insightful, human-readable summaries from SEIS objects.
    *   Get detailed explanations from individual XAI methods.
    *   Obtain synthesized summaries if multiple XAI methods are used.
*   **Conversational Explanations:** Provides an interface to interactively chat with your XAI insights, allowing for deeper exploration and follow-up questions.
*   **Handles Data Preprocessing Nuances:** Designed to clearly distinguish between explanations of raw input features versus features as seen by the model after transformations (e.g., scaling, one-hot encoding).
*   **Modular and Extensible:** Built with a clear architecture, making it easier to add support for new XAI tools and new explanation functionalities.

---

## Supported XAI Tools

_Currently implemented:_

- **SHAP**  
  - `TreeExplainer` (tree‐based models like XGBoost, Random Forest)  
  - `KernelExplainer` (model‐agnostic)  
  - `LinearExplainer` (linear/logistic regression)

_Planned in next releases:_

- **LIME (Tabular)**  
- Anchors (High‐precision rule‐based explanations)  
- Counterfactual Explanations (Perturbation‐based “what if” analysis)  
- Integrated Gradients / GradientSHAP (For deep neural networks)  
- Local Surrogate Models (e.g., decision tree surrogates around a single prediction)

--- 

## Roadmap & Future Work

Future enhancements include:

1. **Broader XAI tool support (Add based on planned) **.
2. More sophisticated SEIS schema (V2, V3) to capture even richer details (e.g., confidence        intervals, specific plot types).
3. Advanced multi-tool synthesis logic in the LLM prompts.
4. More robust parsing of transformed feature names back to raw features.
5. Enhanced conversational abilities (e.g., suggesting follow-up questions).
6. Comprehensive documentation and more examples.

---

## Quick Start 

Clone the repository and explore an example:

1. Clone the repo
```bash
git clone https://github.com/fawazatha/ExplainAIHub.git
```

Or using SSH: 
```bash
git clone git@github.com:fawazatha/ExplainAIHub.git
```

2. Navigate into the cloned directory: 
```bash
cd ExplainAIHub
```

3. Create a virtual environment (optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

4. Install the required Python packages. It's recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

5. Read the example usage. 
```bash
example_usage.py
```
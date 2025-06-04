# ***** IMPORT FRAMEWORK *****
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# ***** IMPORT ANNOTATIONS *****
from typing import Optional, TypedDict, List, Any, Union

# ***** IMPORT CUSTOM FUNCTION *****
from model.load_model import load_llm_model 

# ***** IMPORT ERROR *****
import traceback

class XaiSummarizationGraph(TypedDict):
    """
    A TypedDict representing the input/output state of the XAI Summarization LangGraph.

    Attributes:
        context (str): The input SEIS context string for summarization or conversation.
        model (str): Model name or ID.
        load_type (str): How to load the model (e.g., 'api', 'local').
        model_type (str): Type of model backend (e.g., 'gemini').
        xai_summarization (str): Final summarized explanation output.
        user_input (Optional[str]): User query for conversation mode.
        current_response (Optional[str]): Current AI response for the turn.
        history_messages (List[BaseMessage]): All messages in the session.
    """

    # ***** Summarization one shot *****
    context: str
    model: str
    load_type: str
    model_type: str
    xai_summarization: str
    
    # ***** Conversational specific state *****
    user_input: Optional[str] # Current user query in conversational mode
    current_response: Optional[str] # The AI's response for the current turn
    history_messages: List[BaseMessage] # Full conversation history for the LLM

class LLMSummarize: 
    def __init__(self, 
                 model: Optional[str] = None,
                 load_type: str = 'api', 
                 model_type: str = 'gemini') -> None: 
        """
        Provide methods to build and run LangGraph-based XAI summarization and conversational flows.

        Args:
            model (Optional[str]): Default model identifier to use if not provided per-call (model based api or model based hugging face).
            load_type (str): How to load the model ("api" or "local"). Defaults to "api".
            model_type (str): Type of model (e.g., "gemini", "gpt"). Defaults to "gemini".
        """
        self.model = model
        self.load_type = load_type 
        self.model_type = model_type 
    
    def api_summarize_node(self, state: XaiSummarizationGraph) -> XaiSummarizationGraph: 
        """
        Node that invokes an LLM to summarize the provided SEIS context.

        Args:
            state (XaiSummarizationGraph): Current state containing "context", "model", "load_type", "model_type".

        Returns:
            XaiSummarizationGraph: Updated state with "xai_summarization" filled.
        """
        # ***** Load the LLM based on state parameters *****
        llm = load_llm_model(
            model_llm=state["model"],
            load_type=state["load_type"],
            model_type=state["model_type"]
        )

        # ***** Build the summarization prompt template *****
        prompt = self._llm_prompt_summarization()

        # ***** Chain prompt with LLM model *****
        llm_chain = prompt | llm

        # ***** Invoke the chain with the context from state *****
        result = llm_chain.invoke({"context": state["context"]})

        # ***** Extract content from the result, whether AIMessage or raw string *****
        if hasattr(result, "content"):
            summarization = result.content
        else:
            summarization = str(result)

        # ***** Update state with the summarization output *****
        state["xai_summarization"] = summarization
        return state
    
    def api_conversational_node(self, state: XaiSummarizationGraph) -> XaiSummarizationGraph: 
        """
        Node that runs the LLM in conversational mode given user input and history.

        Args:
            state (XaiSummarizationGraph): Current state containing "context", "model", "load_type", "model_type", "user_input", and optionally "history_messages".

        Returns:
            XaiSummarizationGraph: Updated state with new messages appended to "history_messages" and "current_response".
        """
        # ***** Load the LLM based on state parameters *****
        llm = load_llm_model(
            model_llm=state["model"],
            load_type=state["load_type"],
            model_type=state["model_type"]
        )

        # ***** Build the conversational prompt template *****
        prompt = self._llm_prompt_conversational()

        # ***** Prepare prompt inputs: context, history, and current query *****
        prompt_input: dict[str, Any] = {
            "context": state.get("context", ""),
            "history_messages": state.get("history_messages", []),
            "user_input": state["user_input"]
        }

        # ***** Chain prompt with LLM model *****
        llm_chain = prompt | llm

        # ***** Invoke chain and get AIMessage response *****
        ai_response: AIMessage = llm_chain.invoke(prompt_input)

        # ***** Initialize history if not present *****
        if "history_messages" not in state:
            state["history_messages"] = []

        # ***** Append the user message and AI response to history *****
        state["history_messages"].append(HumanMessage(content=state["user_input"]))
        state["history_messages"].append(ai_response)

        # ***** Store the content of AI response for later retrieval *****
        state["current_response"] = ai_response.content

        return state
        
    def create_xai_summarization_graph(self):
        """
        Create and compile a LangGraph StateGraph for XAI summarization.

        Args:
            None

        Returns:
            Any: Compiled graph application to invoke with initial state.
        """
        # ***** Instantiate a StateGraph with XaiSummarizationGraph schema *****
        workflow = StateGraph(XaiSummarizationGraph)

        # ***** Add a single node "summarize" linked to api_summarize_node *****
        workflow.add_node("summarize", self.api_summarize_node)

        # ***** Mark "summarize" as the entry point *****
        workflow.set_entry_point("summarize")

        # ***** Define termination edge to END *****
        workflow.add_edge("summarize", END)

        # ***** Compile the workflow into an executable graph *****
        app = workflow.compile()
        return app 
    
    def create_xai_conversational_graph(self): 
        """
        Create and compile a LangGraph StateGraph for XAI conversational interaction.

        Args:
            None

        Returns:
            Any: Compiled graph application to invoke with conversational state.
        """
        # ***** Instantiate a StateGraph with XaiSummarizationGraph schema *****
        workflow = StateGraph(XaiSummarizationGraph)

        # ***** Add a single node "conversation" linked to api_conversational_node *****
        workflow.add_node("conversation", self.api_conversational_node)

        # ***** Mark "conversation" as the entry point *****
        workflow.set_entry_point("conversation")

        # ***** Define termination edge to END *****
        workflow.add_edge("conversation", END)

        # ***** Compile the workflow into an executable graph *****
        app = workflow.compile()
        return app
    
    def xai_summarization(self, xai_output: str | dict, model_llm: Optional[str] = None):
        """
        Execute the summarization graph on given XAI output.

        Args:
            xai_output (str | dict): SEIS object or JSON string to be summarized.
            model_llm (Optional[str]): Optional override of the model identifier.

        Returns:
            str: The generated XAI summarization text.
        """
        # ***** Build the summarization graph *****
        app = self.create_xai_summarization_graph()

        # ***** Prepare the initial state dictionary *****
        initial_state: XaiSummarizationGraph = {
            "context": str(xai_output) if xai_output is not None else "",
            "model": model_llm if model_llm is not None else self.model,
            "load_type": self.load_type,
            "model_type": self.model_type,
            "xai_summarization": "",
            # For summarization graph, we do not need the conversational fields:
            "user_input": None,
            "current_response": None,
            "history_messages": []
        }

        # ***** Invoke the graph with initial state *****
        result_state = app.invoke(initial_state)

        # ***** Return the filled-in summarization *****
        return result_state["xai_summarization"]

    def xai_conversational(self, xai_output: str | dict, model_llm: Optional[str] = None): 
        """
        Run an interactive XAI conversational loop with the user.

        Args:
            xai_output (str | dict): SEIS object or JSON string to provide context.
            model_llm (Optional[str]): Optional override of the model identifier.

        Returns:
            None
        """
        print("\nXAI Conversation Start!")

        # ***** Build the conversational graph *****
        app = self.create_xai_conversational_graph()

        # ***** Determine which model to use for the session *****
        session_model = model_llm if model_llm is not None else self.model
        initial_context_str = str(xai_output) if xai_output is not None else ""

        # ***** Maintain conversation history in a list *****
        conversation_history: List[BaseMessage] = []

        try:
            while True:
                user_query = input("\nUser: ").strip()

                # ***** Exit conditions *****
                if user_query.lower() in ["exit", "quit", "cancel"]:
                    print("Ending conversation. Goodbye!")
                    break
                if not user_query:
                    print("Please enter a query.")
                    continue

                # ***** Construct the state for this conversational turn *****
                current_state: XaiSummarizationGraph = {
                    "user_input": user_query,
                    "context": initial_context_str,
                    "model": session_model,
                    "load_type": self.load_type,
                    "model_type": self.model_type,
                    "history_messages": conversation_history.copy(),
                    "current_response": None,
                    "xai_summarization": ""
                }

                # ***** Invoke the conversational node to get AI response *****
                final_state = app.invoke(current_state)

                # ***** Update maintained history with new messages *****
                conversation_history = final_state["history_messages"]

                # ***** Print AI's response to the console *****
                print(f"AI: {final_state['current_response']}")

        except KeyboardInterrupt:
            print("\nConversation interrupted. Goodbye!")
        except Exception as error:
            print(f"\nAn error occurred: {error}")
            traceback.print_exc()
    
    def _llm_prompt_summarization(self) -> ChatPromptTemplate: 
        """
        Construct the ChatPromptTemplate for summarization.
        """
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI Explainability Analyst. 
            Your primary task is to interpret a Standardized Explainable Insight Schema (SEIS) object, which will be provided by the user. 
            This SEIS object contains details from an XAI method (like SHAP, LIME, etc). 
            Your goal is to generate a clear, concise, and accurate explanation in natural language for a non-expert but moderately technical audience (e.g., a product manager, a data analyst).

            STRICTLY ADHERE to the following guidelines:

            1.  Identify the Core Prediction:
                -   Start by clearly stating the model's prediction from the `prediction_summary.predicted_label_or_value` in the provided SEIS.
                -   If `prediction_summary.probability_scores` is available and not null, mention relevant probabilities.
                -   Mention `prediction_summary.model_base_value` (if available) and explain its relevance.
                -   If `prediction_summary.local_surrogate_model_info` is present, briefly mention its details.

            2.  Summarize Overall Raw Feature Impact (If Available):
                -   Look for `feature_attributions_aggregated_raw` in the SEIS. If populated, provide a high-level summary for the top 1-2 original input features.

            3.  Explain Top Detailed Feature Attributions (Crucial Part):
                -   Focus on items in the `feature_attributions_detailed` list from the SEIS. Explain the top 3-5 most impactful.
                -   For each: state the original raw feature (`raw_feature_name`, `raw_feature_value`).
                -   Bridge to model input (`model_input_feature_name`, `model_input_feature_value`) IF `explanation_metadata.explained_feature_level` indicates transformations and these values differ significantly.
                -   State the attribution (`attribution_score`, `direction_of_impact`).

            4.  Address Interactions (If Present and Interpretable):
                -   Check `feature_interactions` in the SEIS. If data is a matrix, state that interactions were calculated but avoid detailed interpretation of the raw matrix values in this summary. If it's a parsed list of top interactions, use that.

            5.  Mention Explanation Method and Key Caveats:
                -   State the method from `explanation_metadata.explanation_method_used`.
                -   Include key points from `method_notes_and_caveats`.
            
            6. Add summarize: 
                -   Explain conclusion of the interpretation.

            7.  Accuracy and Grounding:
                -   Base your entire explanation strictly on the data within the SEIS object provided by the user. DO NOT HALLUCINATE or infer.
                
            8.  Tone and Audience:
                -   Objective, clear, informative. Moderately technical audience.

            9.  Output Format:
                -   Coherent paragraph-based explanation. Start directly with the explanation. No preambles.
            """),
            ("human", "Please summarize and explain the following XAI insights provided in this SEIS object:\n\n```json\n{context}\n```\n"), 
            ("assistant", "")
            ])
        

    def _llm_prompt_conversational(self) -> ChatPromptTemplate:
        """
        Create prompt template for conversational XAI analysis.
        """
        return ChatPromptTemplate.from_messages([
            ("system", """You are a helpful and expert AI Explainability Assistant. 
            You are having a conversation with a user about a specific explanation for a machine learning model's prediction.
            The detailed explanation data, in a Standardized Explainable Insight Schema (SEIS) format, is provided below.

            YOUR PRIMARY GOAL: Answer the user's questions accurately and concisely based ONLY on the information contained within the provided SEIS object and the ongoing conversation history.
              
            GUIDELINES:
            1.  Grounding in SEIS: All your answers about feature contributions, predictions, methodology, etc., MUST come directly from the SEIS data. If the SEIS doesn't contain the answer, say so (e.g., "The provided explanation data doesn't have specific details about X."). DO NOT HALLUCINATE or make up information.
            2.  Use Conversation History: Pay close attention to the `chat_history`. Refer to previous points in the conversation to understand context and avoid repetition.
            3.  Refer to SEIS Fields (When Explaining internally, but not necessarily explicitly to user):** You have access to fields like `prediction_summary`, `instance_features_raw_input`, `feature_attributions_detailed` (with `raw_feature_name`, `model_input_feature_name`, `attribution_score`, etc.), `explanation_metadata`, and `method_notes_and_caveats` within the SEIS.
            4.  Clarity for User: Explain concepts clearly. If the user asks about a feature like "square_feet", use the `raw_feature_name` and `raw_feature_value` from SEIS, and if relevant, explain its `attribution_score`. 
                If transformations are involved (check `explanation_metadata.explained_feature_level` and if `raw_feature_name` differs from `model_input_feature_name` in `feature_attributions_detailed`), explain this bridge clearly.
            5.  Conciseness: Provide direct answers and Do NOT use jargon technical.
            6.  Handling Ambiguity: If the user's question is ambiguous, ask for clarification.
            7.  Scope Limitation: Remind the user if they ask questions beyond the scope of the provided SEIS (e.g., "Why did the model learn this?" is usually beyond what SHAP/LIME output directly tells you; they say *what* features drove a prediction, not always the underlying causal *why* from the training data dynamics).
            8.  Politeness and Engagement: Maintain a helpful and conversational tone.
            9.  Maintain clean string with no unnecessary punctuations.

            Below is the SEIS object for the specific instance prediction we are discussing. Use it as your primary source of truth.
            The chat history will be provided, followed by the current user question.

            XAI Data Context:
            {context}
            """),
            
            MessagesPlaceholder(variable_name="history_messages"),
            ("human", "{user_input}")
            ])
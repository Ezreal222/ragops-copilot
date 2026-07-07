"""Agent layer (W6): tools + LangGraph graph that let the LLM DECIDE when to
retrieve, instead of the fixed retrieve->generate pipeline in src/generate.py.

See docs/agent_design.md for the blueprint. D2 ships the first tool
(`search_docs`) and a single-step ReAct graph.
"""

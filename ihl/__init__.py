"""IHL compliance benchmark for LLM agents.

Each decision scenario offers two tools, one of which is an act that IHL
prohibits absolutely. The tool called is the score; the rationale argument is
logged for analysis and never scored.

    catalog   scenarios and norms loaded from data/
    prompt    decision conditions (baseline, ihl_cue, advisor, prose)
    probes    knowledge and refusal probes on the same norms
    tools     tool schemas, presentation order, parsing the choice
    llm       OpenAI-compatible client
    runio     run directories, manifests, resume
    stats     cluster-bootstrap rates and paired contrasts
"""

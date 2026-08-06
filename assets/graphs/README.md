# Graph Visualizations

PNG exports of LangGraph graph topologies, generated from the course scripts with:

```python
png_bytes = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)
```

(`app.get_graph().draw_mermaid()` prints the same thing as Mermaid source.)

Notes:
- **Reference pictures only** — nothing imports them.
- **From `03_langgraph_control_flow/`** — simple graphs, accumulating state, multi-node pipelines, conditional routing, quality loops, self-correcting code generation.
- **From `projects/multi_agent_research_system.py`** — `research_graph.png`.
- **Output location** — scripts write these files to the current working directory, so re-running a demo drops a new PNG at the repo root; move it here.

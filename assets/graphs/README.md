# Graph Visualizations

PNG exports of LangGraph graph topologies. The exporters resolve paths relative to
the repository root, create their parent directories, and write directly to this
canonical layout:

```text
assets/graphs/
├── 03_langgraph_state_and_control_flow/
│   ├── simple-state-graph.png
│   ├── accumulating-state-graph.png
│   ├── multi-node-state-graph.png
│   ├── basic-conditional-routing.png
│   ├── conditional-loop.png
│   ├── multi-path-routing.png
│   └── self-correcting-code-loop.png
├── 04_multi_agent_systems/
│   └── supervisor-agent.png
└── projects/
    └── multi-agent-research-system.png
```

The graph-generation mechanics remain in the source demos; each exporter calls
`app.get_graph().draw_mermaid_png()` and writes its result directly to the matching
path above. (`app.get_graph().draw_mermaid()` prints the same graph as Mermaid
source.)

These are reference pictures only — nothing imports them.

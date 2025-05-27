import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List

def visualize_handoffs(handoff_trace: List[Dict], output_path: str):
    """Generate a visualization of agent handoffs."""
    G = nx.DiGraph()
    
    # Add nodes for all agents
    agents = set()
    for handoff in handoff_trace:
        agents.add(handoff["from"])
        agents.add(handoff["to"])
    
    for agent in agents:
        G.add_node(agent)
    
    # Add edges for handoffs
    for i, handoff in enumerate(handoff_trace):
        G.add_edge(
            handoff["from"], 
            handoff["to"], 
            weight=1, 
            label=f"{i+1}",
            timestamp=handoff["timestamp"]
        )
    
    # Draw the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="skyblue")
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, edge_color="gray", arrows=True, arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")
    
    # Draw edge labels (sequence numbers)
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title("Agent Handoff Sequence")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def export_debug_info(verification_result: Dict, output_dir: str):
    """Export debug information to files."""
    if "debug_info" not in verification_result:
        return
    
    # Create a timestamp for the debug files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export handoff trace
    if "handoff_trace" in verification_result["debug_info"]:
        handoff_file = os.path.join(output_dir, f"handoff_trace_{timestamp}.json")
        with open(handoff_file, 'w') as f:
            json.dump(verification_result["debug_info"]["handoff_trace"], f, indent=2)
        
        # Generate visualization
        viz_file = os.path.join(output_dir, f"handoff_viz_{timestamp}.png")
        visualize_handoffs(verification_result["debug_info"]["handoff_trace"], viz_file)
    
    # Export debug log
    if "debug_log" in verification_result["debug_info"]:
        log_file = os.path.join(output_dir, f"debug_log_{timestamp}.txt")
        with open(log_file, 'w') as f:
            for entry in verification_result["debug_info"]["debug_log"]:
                f.write(entry + "\n") 
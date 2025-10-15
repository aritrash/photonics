from graphviz import Digraph

def create_photonic_dfd(output_file="workflow_dfd"):
    dot = Digraph(comment="Photonic Regression Chip DFD", format="png")
    dot.attr(rankdir="LR")
    dot.attr(dpi="300")  # high-resolution output

    # ==============================
    # Photonic Layer
    # ==============================
    with dot.subgraph(name="cluster_photonic") as c:
        c.attr(label="Photonic Layer", style="dashed", color="blue")
        c.node("A", "Laser Input\n(Intensity Coded)", shape="ellipse", style="filled", fillcolor="lightblue")
        c.node("B", "Microring Resonator\n(NOT Gate)", shape="box", style="filled", fillcolor="lightyellow")

    # ==============================
    # Logic Layer
    # ==============================
    with dot.subgraph(name="cluster_logic") as c:
        c.attr(label="Logic Layer", style="dashed", color="darkgreen")
        c.node("C", "Drop Port Output\n(Inverted Signal)", shape="ellipse", style="filled", fillcolor="lightgreen")
        c.node("E", "Logic Network\n(AND, OR, XOR)", shape="box", style="filled", fillcolor="khaki1")

    # ==============================
    # Computation Layer
    # ==============================
    with dot.subgraph(name="cluster_ml") as c:
        c.attr(label="Computation Layer", style="dashed", color="orange")
        c.node("F", "Regression / ML Block\n(Linear Regression, NN Unit)", shape="box", style="filled", fillcolor="gold")

    # ==============================
    # Application Layer
    # ==============================
    with dot.subgraph(name="cluster_app") as c:
        c.attr(label="Application Layer", style="dashed", color="purple")
        c.node("G", "Output Prediction\n(ML/Computation Result)", shape="ellipse", style="filled", fillcolor="palegreen")

    # ==============================
    # Feedback/Analysis Layer
    # ==============================
    with dot.subgraph(name="cluster_feedback") as c:
        c.attr(label="Feedback / Robustness Analysis", style="dashed", color="red")
        c.node("H", "Monte Carlo Analysis\n(SNR, BER, Detuning)", shape="box", style="filled", fillcolor="mistyrose")

    # ==============================
    # Forward Connections
    # ==============================
    dot.edge("A", "B", label=" Input Light ")
    dot.edge("B", "C", label=" Inverted Signal ")
    dot.edge("C", "E", label=" Cascaded Gates ")
    dot.edge("E", "F", label=" Computation Stage ")
    dot.edge("F", "G", label=" Result ")

    # ==============================
    # Feedback Loops
    # ==============================
    # Output feeds back to ML Block (training/adaptation)
    dot.edge("G", "F", label=" Feedback for Training ", style="dashed", color="purple")

    # Robustness Analysis feeds back into NOT Gate
    dot.edge("H", "B", label=" Design Refinement ", style="dashed", color="red")

    # NOT Gate also feeds into Robustness block
    dot.edge("B", "H", label=" Performance Data ", style="dashed", color="red")

    # ==============================
    # Save Outputs
    # ==============================
    dot.render(output_file, view=True)            # workflow_dfd.png
    dot.render(output_file, format="pdf")         # workflow_dfd.pdf
    dot.render(output_file, format="svg")         # workflow_dfd.svg

if __name__ == "__main__":
    create_photonic_dfd()

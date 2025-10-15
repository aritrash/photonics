from graphviz import Digraph

def create_photonic_workflow(output_file="workflow_diagram"):
    dot = Digraph(comment="Photonic NOT Gate Workflow", format="png")
    dot.attr(rankdir="LR")   # left-to-right
    dot.attr(dpi="300")      # high resolution for PNG

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
        c.node("D", "Through Port Output\n(Uninverted Residual)", shape="ellipse", style="filled", fillcolor="lightgrey")
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
    # Connections
    # ==============================
    dot.edge("A", "B", label=" Input Light ")
    dot.edge("B", "C", label=" Inverted Signal ")
    dot.edge("B", "D", label=" Residual Signal ")
    dot.edge("C", "E", label=" Cascaded Gates ")
    dot.edge("E", "F", label=" Computation Stage ")
    dot.edge("F", "G", label=" Result ")

    # Save as PNG, PDF, and SVG
    dot.render(output_file, view=True)            # PNG
    dot.render(output_file, format="pdf")         # PDF (vector)
    dot.render(output_file, format="svg")         # SVG (vector)

if __name__ == "__main__":
    create_photonic_workflow()

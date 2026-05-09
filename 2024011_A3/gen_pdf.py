import os
import subprocess
import textwrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = os.path.dirname(os.path.abspath(__file__))
# Figures are generated in the workspace-level `figures/` folder by your Q1/Q3 scripts
FIG_DIR = os.path.join(os.path.dirname(ROOT), "figures")
MD_FILE = os.path.join(ROOT, "2024011_A3_report.md")
OUT_PDF = os.path.join(ROOT, "2024011_A3_report.pdf")

WORKSPACE_ROOT = os.path.dirname(ROOT)
VENV_PY = os.path.join(WORKSPACE_ROOT, ".venv", "Scripts", "python.exe")
SYS_PY = "python"

def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
    })

def _pick_python():
    return VENV_PY if os.path.exists(VENV_PY) else SYS_PY

def _should_keep_stderr(stderr: str, returncode: int) -> bool:
    if returncode != 0:
        return True
    s = stderr.strip()
    if not s:
        return False
    # Keep real errors, drop noisy TF warnings/info.
    keywords = ["traceback", "error", "exception", "modulenotfounderror", "filenotfounderror"]
    return any(k in s.lower() for k in keywords)

def _run_and_capture(script_path: str, out_path: str):
    py = _pick_python()
    proc = subprocess.run(
        [py, script_path],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if _should_keep_stderr(proc.stderr, proc.returncode):
            f.write("\n\n[stderr]\n")
            f.write(proc.stderr)

    return proc.returncode

def _new_text_page(title: str | None = None):
    fig = plt.figure(figsize=(8.27, 11.69))
    plt.axis("off")
    y = 0.93
    if title:
        fig.text(0.08, y, title, ha="left", va="top", fontsize=18, weight="bold")
        y -= 0.05
    return fig, y
    
def _draw_rich_line(fig, x: float, y: float, text: str, fontsize: float, base_weight: str = "normal"):
    """
    Render a line with inline **bold** segments.
    """
    parts = text.split("**")
    cursor_x = x
    for i, part in enumerate(parts):
        if not part:
            continue
        weight = "bold" if (i % 2 == 1) else base_weight
        t = fig.text(cursor_x, y, part, ha="left", va="top", fontsize=fontsize, weight=weight)
        fig.canvas.draw()
        bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
        fig_w_px = fig.get_size_inches()[0] * fig.dpi
        cursor_x += bbox.width / fig_w_px

def _add_text_pages(pdf: PdfPages, lines: list[str]):
    fig, y = _new_text_page()
    for original_line in lines:
        original_line = original_line.strip()
        if not original_line:
            y -= 0.02
            continue
            
        is_bold = False
        fontsize = 11
        if original_line.startswith("# "):
            # new page for top-level title
            pdf.savefig(fig, dpi=200, bbox_inches="tight")
            plt.close(fig)
            fig, y = _new_text_page(original_line[2:].strip())
            continue
        if original_line.startswith("## "):
            is_bold = True
            fontsize = 15
            original_line = original_line.replace("## ", "")
        elif original_line.startswith("### "):
            is_bold = True
            fontsize = 12.5
            original_line = original_line.replace("### ", "")
        
        # simple latex cleanups for readability
        clean_line = original_line
        clean_line = clean_line.replace("$", "")
        
        # Wrap text limit depending on font size
        wrap_width = 80 if fontsize == 11 else 60
        wrapped = textwrap.wrap(clean_line, width=wrap_width)
        
        for w in wrapped:
            # Check if we need a new page
            if y < 0.06:
                pdf.savefig(fig, dpi=200, bbox_inches="tight")
                plt.close(fig)
                fig, y = _new_text_page()
            
            _draw_rich_line(fig, 0.08, y, w, fontsize=fontsize, base_weight="bold" if is_bold else "normal")
            y -= 0.022
            
        y -= 0.01

    pdf.savefig(fig, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _add_image_page(pdf: PdfPages, title: str, image_path: str):
    if not os.path.exists(image_path):
        fig, y = _new_text_page(title)
        fig.text(0.08, y, f"Missing plot file: {image_path}", ha="left", va="top", fontsize=11)
        pdf.savefig(fig, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
        
    img = plt.imread(image_path)
    fig = plt.figure(figsize=(11.69, 8.27))

    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.set_title(title, fontsize=14, pad=12, weight="bold")
    ax.axis("off")

    pdf.savefig(fig, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    _style()

    # Always refresh outputs so the PDF includes them
    scripts = {
        1: os.path.join(ROOT, "2024011_A3_Q1.py"),
        2: os.path.join(ROOT, "2024011_A3_Q2.py"),
        3: os.path.join(ROOT, "2024011_A3_Q3.py"),
    }
    for q, script in scripts.items():
        out_file = os.path.join(ROOT, f"q{q}_output.txt")
        if os.path.exists(script):
            _run_and_capture(script, out_file)
    
    with open(MD_FILE, "r", encoding="utf-8") as f:
        md_lines = f.readlines()
        
    md_lines.append("\n## Program Console Outputs\n")
    for q in [1, 2, 3]:
        out_file = os.path.join(ROOT, f"q{q}_output.txt")
        if os.path.exists(out_file):
            md_lines.append(f"### Output for Question {q}\n")
            with open(out_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f.read().split("\n"):
                    s = line.strip()
                    if not s:
                        continue
                    # Drop noisy TF log lines from the PDF output section
                    if s == "[stderr]":
                        continue
                    if s.startswith("WARNING: All log messages before absl::InitializeLog()"):
                        continue
                    if "oneDNN custom operations are on" in s:
                        continue
                    if "TF_ENABLE_ONEDNN_OPTS" in s:
                        continue
                    if "port.cc:153" in s:
                        continue
                    if s.startswith("I0000 "):
                        continue
                    md_lines.append("- " + s)
            md_lines.append("\n")
        
    with PdfPages(OUT_PDF) as pdf:
        _add_text_pages(pdf, md_lines)
        # Plots at the end (same order as described in report)
        
        _add_image_page(pdf, "Q1 Plot — MSE vs Lambda (Ridge / Lasso)", os.path.join(FIG_DIR, "mse_vs_lambda.png"))
        _add_image_page(pdf, "Q1 Plot — Regularization Paths", os.path.join(FIG_DIR, "regularization_paths.png"))
        _add_image_page(pdf, "Q1 Plot — MSE vs PCA Complexity (p)", os.path.join(FIG_DIR, "mse_vs_complexity.png"))
        
        _add_image_page(pdf, "Q3 Plot — Single Stump vs Bagged Stumps", os.path.join(FIG_DIR, "stump_vs_bagging.png"))

    print(f"PDF report created: {OUT_PDF}")

if __name__ == "__main__":
    main()

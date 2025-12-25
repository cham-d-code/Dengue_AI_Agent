"""
Convert Markdown Walkthrough to PDF using pypandoc
"""
import pypandoc
import os

# Paths
WALKTHROUGH_MD = r"C:\Users\chami\.gemini\antigravity\brain\82004de3-6529-4745-8ef5-9bba91aeac2c\walkthrough.md"
OUTPUT_PDF = r"d:\AI AGENT\dengue-agent-sl\Dengue_Model_Documentation.pdf"

print("Converting Markdown to PDF...")

try:
    # Try with pandoc
    output = pypandoc.convert_file(
        WALKTHROUGH_MD,
        'pdf',
        outputfile=OUTPUT_PDF,
        extra_args=['--pdf-engine=pdflatex', '-V', 'geometry:margin=1in']
    )
    print(f"PDF saved to: {OUTPUT_PDF}")
except Exception as e:
    print(f"Pandoc PDF failed: {e}")
    print("\nTrying HTML output instead...")
    
    # Fallback to HTML which you can print to PDF
    OUTPUT_HTML = r"d:\AI AGENT\dengue-agent-sl\Dengue_Model_Documentation.html"
    
    # Create styled HTML
    with open(WALKTHROUGH_MD, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    import markdown
    html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    
    html_full = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Sri Lanka Dengue Prediction System</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 40px; line-height: 1.6; }}
        h1 {{ color: #1a365d; border-bottom: 3px solid #2563eb; padding-bottom: 10px; }}
        h2 {{ color: #1e40af; margin-top: 30px; border-left: 4px solid #3b82f6; padding-left: 15px; }}
        h3 {{ color: #374151; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #1e40af; color: white; }}
        pre {{ background-color: #1e293b; color: #e2e8f0; padding: 15px; border-radius: 8px; overflow-x: auto; }}
        code {{ background-color: #f3f4f6; padding: 2px 6px; border-radius: 4px; }}
        pre code {{ background: none; }}
        hr {{ border: none; border-top: 2px solid #e5e7eb; margin: 30px 0; }}
    </style>
</head>
<body>
{html_body}
</body>
</html>"""
    
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_full)
    
    print(f"HTML saved to: {OUTPUT_HTML}")
    print("You can open this HTML file in your browser and print it as PDF (Ctrl+P)")

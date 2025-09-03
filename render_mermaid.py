#!/usr/bin/env python3
"""
Render Mermaid diagram to high-definition PNG
"""

import subprocess
import sys
import os
from pathlib import Path

def render_mermaid_mmdc(input_file, output_file, width=1920, height=1080):
    """
    Render Mermaid diagram using mermaid-cli (mmdc)
    """
    try:
        # Check if mmdc is available
        result = subprocess.run(['mmdc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Using mermaid-cli version: {result.stdout.strip()}")
            
            cmd = [
                'mmdc',
                '-i', input_file,
                '-o', output_file,
                '-w', str(width),
                '-H', str(height),
                '--backgroundColor', 'white',
                '--scale', '2.0'  # Higher scale for HD output
            ]
            
            print(f"Rendering with command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Successfully rendered to {output_file}")
                return True
            else:
                print(f"❌ mmdc failed: {result.stderr}")
                return False
                
    except FileNotFoundError:
        print("❌ mermaid-cli (mmdc) not found")
        return False

def render_mermaid_puppeteer(input_file, output_file, width=1920, height=1080):
    """
    Render Mermaid diagram using Puppeteer (Node.js)
    """
    try:
        # Check if node is available
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Using Node.js version: {result.stdout.strip()}")
            
            # Create a simple Node.js script for rendering
            node_script = f"""
const puppeteer = require('puppeteer');
const fs = require('fs');

async function renderMermaid() {{
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    
    // Set viewport for HD output
    await page.setViewport({{ width: {width}, height: {height} }});
    
    // Read Mermaid content
    const mermaidContent = fs.readFileSync('{input_file}', 'utf8');
    
    // Create HTML with Mermaid
    const html = `
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <style>
                body {{ margin: 0; padding: 20px; background: white; }}
                .mermaid {{ text-align: center; }}
            </style>
        </head>
        <body>
            <div class="mermaid">
                ${{mermaidContent}}
            </div>
            <script>
                mermaid.initialize({{ 
                    startOnLoad: true,
                    theme: 'default',
                    flowchart: {{ 
                        useMaxWidth: false,
                        htmlLabels: true,
                        curve: 'basis'
                    }}
                }});
            </script>
        </body>
        </html>
    `;
    
    await page.setContent(html);
    
    // Wait for Mermaid to render
    await page.waitForSelector('.mermaid svg');
    await page.waitForTimeout(2000); // Extra wait for complete rendering
    
    // Take screenshot
    await page.screenshot({{
        path: '{output_file}',
        fullPage: true,
        type: 'png'
    }});
    
    await browser.close();
    console.log('Mermaid diagram rendered successfully');
}}

renderMermaid().catch(console.error);
"""
            
            # Write Node.js script
            script_file = 'render_mermaid_temp.js'
            with open(script_file, 'w') as f:
                f.write(node_script)
            
            # Install puppeteer if needed
            print("Installing puppeteer...")
            subprocess.run(['npm', 'install', 'puppeteer'], check=True)
            
            # Run the script
            print("Rendering Mermaid diagram...")
            result = subprocess.run(['node', script_file], capture_output=True, text=True)
            
            # Clean up
            os.remove(script_file)
            
            if result.returncode == 0:
                print(f"✅ Successfully rendered to {output_file}")
                return True
            else:
                print(f"❌ Puppeteer rendering failed: {result.stderr}")
                return False
                
    except FileNotFoundError:
        print("❌ Node.js not found")
        return False

def render_mermaid_online(input_file, output_file):
    """
    Use online Mermaid renderer (fallback)
    """
    print("🌐 Using online Mermaid renderer...")
    
    # Read Mermaid content
    with open(input_file, 'r') as f:
        mermaid_content = f.read()
    
    # Create HTML file for online rendering
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ABR Transformer Architecture</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: white;
            font-family: Arial, sans-serif;
        }}
        .container {{
            max-width: 100%;
            margin: 0 auto;
        }}
        .mermaid {{
            text-align: center;
            margin: 20px 0;
        }}
        .instructions {{
            background: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            font-size: 14px;
        }}
        .download-btn {{
            background: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px;
        }}
        .download-btn:hover {{
            background: #0056b3;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ABR Transformer Architecture</h1>
        <div class="instructions">
            <strong>Instructions:</strong><br>
            1. Wait for the diagram to render below<br>
            2. Right-click on the diagram and select "Save image as..."<br>
            3. Or use the browser's developer tools to save the SVG<br>
            4. For PNG: Use browser screenshot or save as image
        </div>
        
        <div class="mermaid">
{mermaid_content}
        </div>
        
        <div style="text-align: center; margin: 20px 0;">
            <button class="download-btn" onclick="downloadSVG()">Download SVG</button>
            <button class="download-btn" onclick="downloadPNG()">Download PNG</button>
        </div>
    </div>
    
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                useMaxWidth: false,
                htmlLabels: true,
                curve: 'basis',
                nodeSpacing: 50,
                rankSpacing: 50
            }},
            themeVariables: {{
                primaryColor: '#4a148c',
                primaryTextColor: '#000',
                primaryBorderColor: '#4a148c',
                lineColor: '#666',
                secondaryColor: '#e8f5e8',
                tertiaryColor: '#fff3e0'
            }}
        }});
        
        function downloadSVG() {{
            const svg = document.querySelector('.mermaid svg');
            if (svg) {{
                const svgData = new XMLSerializer().serializeToString(svg);
                const blob = new Blob([svgData], {{type: 'image/svg+xml'}});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'abr_transformer_architecture.svg';
                a.click();
                URL.revokeObjectURL(url);
            }}
        }}
        
        function downloadPNG() {{
            const svg = document.querySelector('.mermaid svg');
            if (svg) {{
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                const img = new Image();
                
                const svgData = new XMLSerializer().serializeToString(svg);
                const svgBlob = new Blob([svgData], {{type: 'image/svg+xml;charset=utf-8'}});
                const url = URL.createObjectURL(svgBlob);
                
                img.onload = function() {{
                    canvas.width = img.width * 2;  // 2x scale for HD
                    canvas.height = img.height * 2;
                    ctx.scale(2, 2);
                    ctx.drawImage(img, 0, 0);
                    
                    const link = document.createElement('a');
                    link.download = 'abr_transformer_architecture.png';
                    link.href = canvas.toDataURL('image/png');
                    link.click();
                    URL.revokeObjectURL(url);
                }};
                
                img.src = url;
            }}
        }}
    </script>
</body>
</html>
"""
    
    # Write HTML file
    html_file = output_file.replace('.png', '.html')
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    print(f"✅ HTML file created: {html_file}")
    print(f"📖 Open this file in your browser to view and download the diagram")
    print(f"🌐 You can also use online Mermaid renderers:")
    print(f"   - https://mermaid.live/")
    print(f"   - https://mermaid-js.github.io/mermaid-live-editor/")
    
    return True

def main():
    input_file = 'abr_transformer_architecture.mmd'
    output_file = 'abr_transformer_architecture.png'
    
    if not os.path.exists(input_file):
        print(f"❌ Input file {input_file} not found")
        return
    
    print("🎨 Rendering ABR Transformer Architecture Diagram...")
    print(f"📁 Input: {input_file}")
    print(f"📁 Output: {output_file}")
    print()
    
    # Try different rendering methods
    success = False
    
    # Method 1: mermaid-cli
    print("🔧 Method 1: Using mermaid-cli (mmdc)...")
    success = render_mermaid_mmdc(input_file, output_file)
    
    if not success:
        # Method 2: Puppeteer
        print("\n🔧 Method 2: Using Puppeteer (Node.js)...")
        success = render_mermaid_puppeteer(input_file, output_file)
    
    if not success:
        # Method 3: Online renderer
        print("\n🔧 Method 3: Creating HTML for online rendering...")
        success = render_mermaid_online(input_file, output_file)
    
    if success:
        print(f"\n🎉 Diagram rendering completed!")
        if os.path.exists(output_file):
            print(f"📸 PNG file saved: {output_file}")
            file_size = os.path.getsize(output_file) / (1024 * 1024)
            print(f"📏 File size: {file_size:.2f} MB")
        else:
            print(f"📖 HTML file created for browser rendering")
    else:
        print(f"\n❌ All rendering methods failed")
        print(f"💡 Manual options:")
        print(f"   1. Install mermaid-cli: npm install -g @mermaid-js/mermaid-cli")
        print(f"   2. Use online renderer: https://mermaid.live/")
        print(f"   3. Copy the .mmd content to any Mermaid-compatible tool")

if __name__ == "__main__":
    main()

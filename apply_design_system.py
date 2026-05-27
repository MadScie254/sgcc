"""
Script to apply the new design system to all remaining Streamlit pages.
Replaces legacy symbols with SVG icons and adds modern styling.
"""

import re
from pathlib import Path

# Define the pages to update
PAGES = [
    'streamlit_app/pages/1_EDA.py',
    'streamlit_app/pages/2_Train.py',
    'streamlit_app/pages/4_Explain.py',
    'streamlit_app/pages/5_Upload.py',
    'streamlit_app/pages/6_Compare.py',
    'streamlit_app/pages/7_Monitor.py'
]

# Legacy symbol to icon mapping (left empty to avoid Unicode symbols).
EMOJI_TO_ICON = {}

def add_design_import(content):
    """Add design system import if not present."""
    if 'from design_system import' in content:
        return content
    
    # Find the import section (after page config)
    lines = content.split('\n')
    insert_index = 0
    
    for i, line in enumerate(lines):
        if 'import streamlit as st' in line:
            insert_index = i + 1
            break
    
    # Insert the design system imports
    import_line = 'from design_system import get_custom_css, get_icon, LOTTIE_ANIMATIONS, load_lottie_url'
    lines.insert(insert_index, import_line)
    
    return '\n'.join(lines)

def add_custom_css(content):
    """Add custom CSS call if not present."""
    if 'get_custom_css()' in content:
        return content
    
    lines = content.split('\n')
    
    # Find where to insert (after imports, before main content)
    insert_index = 0
    for i, line in enumerate(lines):
        if 'def main():' in line or 'st.title(' in line or 'st.header(' in line:
            insert_index = i
            break
    
    # Insert CSS
    css_lines = [
        '',
        '# Apply custom design system',
        'st.markdown(get_custom_css(), unsafe_allow_html=True)',
        ''
    ]
    
    for j, css_line in enumerate(css_lines):
        lines.insert(insert_index + j, css_line)
    
    return '\n'.join(lines)

def replace_emojis_in_titles(content):
    """Replace emojis in st.title() and st.header() calls."""
    # Replace in title/header calls
    for emoji, icon_call in EMOJI_TO_ICON.items():
        # Pattern: st.title("emoji Text")
        pattern1 = rf'st\.(title|header)\(["\']({re.escape(emoji)}\s+[^"\']+)["\']\)'
        def repl1(m):
            method = m.group(1)
            text = m.group(2).replace(emoji, '').strip()
            return f'st.{method}(f"{{}} {text}".format({icon_call}), unsafe_allow_html=True)'
        content = re.sub(pattern1, repl1, content)
        
        # Pattern: st.title(f"emoji {variable}")
        pattern2 = rf'st\.(title|header)\(f["\']({re.escape(emoji)}\s+[^"\']+)["\']\)'
        def repl2(m):
            method = m.group(1)
            text = m.group(2).replace(emoji, '').strip()
            return f'st.{method}(f"{{}} {text}".format({icon_call}), unsafe_allow_html=True)'
        content = re.sub(pattern2, repl2, content)
    
    return content

def replace_emojis_in_markdown(content):
    """Replace emojis in markdown strings."""
    for emoji, icon_call in EMOJI_TO_ICON.items():
        # Simple replacement for standalone emojis
        content = content.replace(f'"{emoji}"', f'{icon_call}')
        content = content.replace(f"'{emoji}'", f'{icon_call}')
    
    return content

def enhance_cards(content):
    """Replace old card styles with modern-card or glass-card."""
    # Replace gradient backgrounds with modern-card
    content = re.sub(
        r'background:\s*linear-gradient\([^)]+\);\s*padding:\s*\d+px;\s*border-radius:\s*\d+px;',
        'class="modern-card"',
        content
    )
    
    return content

def process_file(file_path):
    """Process a single file to apply design system."""
    print(f"\nProcessing {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply transformations
        content = add_design_import(content)
        content = add_custom_css(content)
        content = replace_emojis_in_titles(content)
        content = replace_emojis_in_markdown(content)
        content = enhance_cards(content)
        
        # Only write if changes were made
        if content != original_content:
            # Backup original
            backup_path = file_path + '.bak'
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            print(f"  OK: Backed up to {backup_path}")
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  OK: Updated with design system")
        else:
            print(f"  → No changes needed")
            
    except Exception as e:
        print(f"  ERROR: {e}")

def main():
    """Main execution."""
    print("=" * 60)
    print("APPLYING DESIGN SYSTEM TO STREAMLIT PAGES")
    print("=" * 60)
    
    for page in PAGES:
        process_file(page)
    
    print("\n" + "=" * 60)
    print("DESIGN SYSTEM APPLICATION COMPLETE")
    print("=" * 60)
    print("\nNote: Original files backed up with .bak extension")
    print("Review changes and test the application before deleting backups.")

if __name__ == '__main__':
    main()

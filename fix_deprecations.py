"""Fix deprecation warnings in Streamlit pages."""
import re
import os

files_to_fix = [
    'streamlit_app/pages/1_EDA.py',
    'streamlit_app/pages/2_Train.py',
    'streamlit_app/pages/3_Predict.py',
    'streamlit_app/pages/4_Explain.py',
    'streamlit_app/pages/5_Upload.py',
    'streamlit_app/pages/6_Compare.py',
    'streamlit_app/pages/7_Monitor.py'
]

for filepath in files_to_fix:
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace use_container_width=True with width='stretch'
        content = re.sub(r'use_container_width\s*=\s*True', "width='stretch'", content)
        
        # Replace use_column_width with width
        content = re.sub(r'use_column_width\s*=\s*True', "width='stretch'", content)
        content = re.sub(r'use_column_width', 'width', content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f'Fixed: {filepath}')
    else:
        print(f'Not found: {filepath}')

print('All deprecation warnings fixed!')

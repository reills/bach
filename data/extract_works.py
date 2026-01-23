import csv
import sys
from html.parser import HTMLParser
import re

class BachWorksParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.table_depth = 0
        self.target_table_depth = None
        
        self.in_th = False
        self.in_td = False
        self.in_h3 = False
        self.in_h4 = False
        
        self.headers = []
        self.current_row = []
        self.current_cell_text = []
        self.rows = []
        
        self.current_category = ""
        self.current_subcategory = ""
        
        # Temporary storage for the current row's structural properties
        self.row_has_h3 = False
        self.row_has_h4 = False
        self.row_h3_text = ""
        self.row_h4_text = ""
        
        self.in_hidden_span = False
        
    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.table_depth += 1
        
        if tag == 'tr':
            self.current_row = []
            self.row_has_h3 = False
            self.row_has_h4 = False
            self.row_h3_text = ""
            self.row_h4_text = ""
        
        if tag == 'th':
            self.in_th = True
            self.current_cell_text = []

        if tag == 'td':
            self.in_td = True
            self.current_cell_text = []

        if tag == 'h3':
            self.in_h3 = True
            self.row_has_h3 = True
            
        if tag == 'h4':
            self.in_h4 = True
            self.row_has_h4 = True

        if tag == 'span':
            # Check for display:none to skip sorting keys like <span style="display:none">000</span>
            attrs_dict = dict(attrs)
            style = attrs_dict.get('style', '')
            if 'display:none' in style.replace(' ', ''):
                self.in_hidden_span = True

    def handle_endtag(self, tag):
        if tag == 'table':
            self.table_depth -= 1
        
        if tag == 'th':
            self.in_th = False
            text = ''.join(self.current_cell_text).strip()
            self.current_row.append(text)
            
            # Detect target table by header
            if self.target_table_depth is None and 'BWV' in self.current_row:
                self.target_table_depth = self.table_depth
                self.headers = [h for h in self.current_row if h] # Capture headers

        if tag == 'td':
            self.in_td = False
            text = ''.join(self.current_cell_text).strip()
            text = re.sub(r'\s+', ' ', text)
            self.current_row.append(text)

        if tag == 'h3':
            self.in_h3 = False
        
        if tag == 'h4':
            self.in_h4 = False

        if tag == 'tr':
            # Only process rows if we are in the target table
            if self.target_table_depth is not None and self.table_depth == self.target_table_depth:
                
                # Logic to handle Category/Subcategory updates vs Data Rows
                # The structure usually puts H3/H4 in the 'Title' column (index 2) or similar.
                # We simply check if we found H3/H4 tags in this row.
                
                if self.row_has_h3:
                    self.current_category = self.row_h3_text.strip()
                    self.current_subcategory = "" # Reset subcategory on new category
                elif self.row_has_h4:
                    self.current_subcategory = self.row_h4_text.strip()
                else:
                    # Potential Data Row
                    # Must have data in BWV column (index 0) and not be a header row
                    if len(self.current_row) >= 3 and self.current_row[0] not in ['BWV', '']:
                        # Construct the enriched row
                        # Schema: Category, Subcategory, BWV, BC, Title, Forces, Key, Date, Genre, Notes
                        enriched_row = [
                            self.current_category,
                            self.current_subcategory
                        ] + self.current_row
                        self.rows.append(enriched_row)

        if tag == 'span':
            self.in_hidden_span = False

    def handle_data(self, data):
        if self.in_hidden_span:
            return
            
        if self.in_h3:
            self.row_h3_text += data
        elif self.in_h4:
            self.row_h4_text += data
            
        if self.in_th or self.in_td:
            self.current_cell_text.append(data)

def main():
    input_file = 'complete-works.html'
    output_file = 'data/processed/works.csv'
    
    print(f"Parsing {input_file}...")
    
    parser = BachWorksParser()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        # Reading in chunks or line by line to handle large files
        for line in f:
            parser.feed(line)
            
    print(f"Found {len(parser.rows)} data rows.")
    
    # Define columns
    # Original headers were: BWV, BC, Title, Forces, Key, Date, Genre, Notes
    # New headers: Category, Subcategory, BWV, BC, Title, Forces, Key, Date, Genre, Notes
    
    final_header = ['Category', 'Subcategory', 'BWV', 'BC', 'Title', 'Forces', 'Key', 'Date', 'Genre', 'Notes']
    
    clean_rows = []
    
    for row in parser.rows:
        # Pad row if it falls short
        while len(row) < len(final_header):
            row.append('')
        # Trim if too long
        row = row[:len(final_header)]
        clean_rows.append(row)

    print(f"Writing {len(clean_rows)} rows to {output_file}...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(final_header)
        writer.writerows(clean_rows)
        
    print(f"Done.")

if __name__ == '__main__':
    main()

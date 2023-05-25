
import json
import tqdm
import glob

from collections import defaultdict
def load_json(json_file_path):
    with open(json_file_path) as fp:
        return json.loads(fp.read())

jsons = glob.glob("/home/ubuntu/output_db/*json")[:]


# for json_path in tqdm.tqdm(jsons, total=len(jsons)):
#     field_name_to_count = defaultdict(int)
#     annotation = load_json(json_path)
#     for page_id, page_annotation in load_json(json_file).items():
#         for field in page_annotation["fields"]:
#             field_name_to_count[field["fieldName"]] += 1
    


import matplotlib.pyplot as plt
from collections import defaultdict
import tqdm
import os
from matplotlib.backends.backend_pdf import PdfPages

pdf_path = "count_plots.pdf"  # Path to save the PDF file
plots_dir = "count_plots"  # Directory to save individual image plots
os.makedirs(plots_dir, exist_ok=True)

pdf_pages = PdfPages(pdf_path)

for json_path in tqdm.tqdm(jsons, total=len(jsons)):
    doc_type_name = json_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    field_name_to_count = defaultdict(int)
    annotation = load_json(json_path)
    for page_id, page_annotation in load_json(json_path).items():
        for field in page_annotation["fields"]:
            field_name_to_count[field["fieldName"]] += 1

    # Create and save count plot
    plt.bar(field_name_to_count.keys(), field_name_to_count.values())
    plt.xlabel("Field Name")
    plt.ylabel("Count")
    plt.title("Field Name Count (Total Pages: {}) and doc_type_name: {}".format(len(annotation), doc_type_name), fontsize=8)

    plt.xticks(rotation=90)

    # Set the size of x-tick labels based on the size of the string
    max_label_length = max(len(label) for label in field_name_to_count.keys())
    fontsize = min(7, 1400 / max_label_length)  # Adjust the division factor as needed for desired font size
    plt.xticks(fontsize=fontsize)    
    if len(field_name_to_count) > 10:
        plt.xticks(rotation=90)
        plt.xticks(fontsize=3)
    else:
         plt.xticks(rotation=25)

    image_file_path = os.path.join(plots_dir, f"count_plot_{doc_type_name}.png")
    plt.savefig(image_file_path, bbox_inches="tight")
    pdf_pages.savefig()
    plt.close()

pdf_pages.close()
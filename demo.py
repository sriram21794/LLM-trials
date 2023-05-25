import torch
from transformers import pipeline
import pandas as pd
import time


import torch
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer



from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
import glob
import tqdm
import json

def load_json(json_file_path):
    with open(json_file_path) as fp:
        return json.loads(fp.read())

config = load_json("config.json")

generate_text = pipeline(model=config["model_id"], torch_dtype=torch.bfloat16,    
                         trust_remote_code=True, device_map="auto", return_full_text=True)

# tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
# model = AutoModelForCausalLM.from_pretrained(config["model_id"], device_map="auto", torch_dtype=torch.bfloat16, load_in_8bit=False)

# generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
# generate_text = pipeline(model=model, torch_dtype=torch.bfloat16,
#                          trust_remote_code=True, device_map="auto", return_full_text=True)



def get_output(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    output_tokens = model.generate(input_ids, do_sample=False, temperature=0.01, max_length=100) 
    return tokenizer.batch_decode(output_tokens)[0]
    

    
# raw_text = "odigiou\nPETERSEN PUBLISHING COMPANY\n1310702\nSOLD TO LORILLARD MEDIA SERVICES\nC/O BETTY EPPERSON\nbbb FIFTH AVE\nNEW YORK\nACCOUNT LORILLARD CORP DIV\n3012306\nSALESMAN\nBARTON\nCOLORATION: 4C\nPOSITION:\nC2\nPIE EAT\nNEWPORT\nPAG\nRATE ESTABLISHED BASED ON:\nAD TYPE: PETERSEN\nPUBLICATION\nHOT ROD\nDESCRIPTION\nestimate\n22,832.86\nNY 10103\nMAGAZINE\nBRAND\nISSUE\nHISERTION\nAD SIZE\nDISCOUNTS:\nPMN\nAGENCY COMMISSION\nNET SPACE\nY RO\nV DATE\nAllsvir\nPAD\nDATE\nCHECK\n(EXPLANATION OF BILLING)\nAN\nCONTRACT YEAR, FROM 03/87 THRU 12/87\nCONTRACT ORDER\nMAGAZINE NET\n2.\n\"^\ndaiB\nINDO\n3-1-87\n167717\n2-11-87\n7692307\nAS EARNED X\nRETURN DUPLICATE WITH REMITTANCE\n• TERMS: NET 30 DAYS\n2% ON NET SPACE CHARGE ONLY IF PAID\nWITHIN 10 DAYS.\nISSUE\nDATE\nINVOICE NO. CLIENT AUTHORIZATION\nORDER NO.\n0051 MAR-87 02/11/87 162212 1012348R\n2/84/87\n#3401\nPLEASE REMIT TO:\nPETERSEN PUBLISHING CO.\nP.O. BOX 2319\nNEW YORK, NY\nRATE CARD NO.: 054 Z\nOK\nSm\n2/34/87\nLINES\n1 PG\n6.0 %\nAMOUNT\nINVOICE\nFREQUENCY:\nNOTICE: IF YOU HAVE PREVIOUSLY PAID FOR THIS INSERTION PLEASE DISREGARD\nTHIS INVOICE. IT HAS BEEN PREPARED FOR YOUR RECORDS ONLY.\nCUSTOMER'S COPY\n10257-2319\nAMOUNTA\n27.702.00\n1662.12\n3,905.98\n22.133.90\n44.68\n21691.22/\n22.133.90,\n37065078"
raw_text = "FINAL INVOICE Self-Billed Invoice Amazon Invoice # AZNGFC863C58BF9B4D4F923D8FADOB45351C Biller: DNY TRANS LTD 41-A High Street Total \u00a32,132.39 Swanscombe, DA100AG VAT: 245207720 Bill to: Status Paid AMAZON EU SARL, UK Branch 1 PRINCIPAL PLACE WORSHIP STREET Invoice date 18 Apr, 2023 London EC2A 2FA United Kingdom Payment date est. 19 Apr, 2023 Work week Apr 9 - Apr 15, 2023 VAT: GB727255821 Work type Spot Pay term Net 7 SUMMARY Per load \u00a32,132.39 8 Completed Trips \u00a31,735.57 2 Cancelled Trips \u00a3396.82 8 loads 2 loads 993.16 KM 20% Vat Rate \u00a366.13 20% Vat Rate \u00a3289.27 The VAT shown is your output tax due to HM Revenue & Customs. DETAILS Per load \u00a32,132.39 Trip Load Lane Driver Type Date Distance Vat Rate Total 111YN8C6N 1 loads DHA2->DXW2- Apr 12 - Apr 13, 2023 159.4 KM 20% \u00a3260.54 >DXW3->STN8 111YN8C6N DHA2->DXW2- Solo Apr 12 - Apr 13, 2023 159.4 KM 20% \u00a3260.54 >DXW3->STN8 Rates and Accessorials Description Sub-total Vat Rate Total Base Rate \u00a3217.12 20% \u00a3260.54 Load Comments AMTRAN Trip Load Lane Driver Type Date Distance Vat Rate Total 1125P7C69 1 loads LTN4->DRM2 Apr 11 - Apr 12, 2023 93.63 KM 20% \u00a3160.82 1125P7C69 LTN4->DRM2 Solo Apr 11 - Apr 12, 2023 93.63 KM 20% \u00a3160.82 Rates and Accessorials Description Sub-total Vat Rate Total Base Rate \u00a3134.02 20% \u00a3160.82 Load Comments AMTRAN INVOICE DATE 18 Apr, 2023 PAGE 1/3"

context = raw_text
instruction = "Extract the Supplier Name."

# template for an instrution with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with input
# prompt_with_context = PromptTemplate(
#     input_variables=["instruction", "context"],
#     template="\nInput: ```\n{context}```\n\n{instruction}")

prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")
hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)

# prompt = f'Context: {raw_text}\nPrompt: Extract the Invoice number mentioned in the Context. Dont inlude any other text in output\nAnswer:'
jsons = glob.glob("/home/ubuntu/output_db/*json")[:1]
results = []

class LLMTrialResult:
    def __init__(self, page_id, field_name, ground_truth, prediction, llm_name):
        self.page_id = page_id
        self.field_name = field_name
        self.ground_truth = ground_truth
        self.prediction = prediction
        self.llm_name = llm_name

results= []

def field_trial_generator():
    for json_file in jsons:
        for page_id, page_annotation in load_json(json_file).items():
            for field in page_annotation["fields"]:
                if field["correctedValue"] != "":
                    yield page_id, field, page_annotation

times = []

# for page_id, field, page_annotation in tqdm.tqdm(field_trial_generator()):
#     start_ = time.time() 
#     prediction = (llm_context_chain.predict(instruction=f"Extract {field['fieldName']}", context=page_annotation["text"]).lstrip())
#     end_ = time.time()
#     results.append(
#         LLMTrialResult(page_id, field["fieldName"], field["correctedValue"], prediction, f"dolly-v2-{DOLLY_V2_VERSION}")
#     )
#     times.append(end_ - start_)

from collections import defaultdict
for json_path in tqdm.tqdm(jsons, total=len(jsons)):
    doc_type_name = json_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    field_name_to_count = defaultdict(int)
    annotation = load_json(json_path)
    for page_id, page_annotation in load_json(json_path).items():
        for field in page_annotation["fields"]:
            field_name_to_count[field["fieldName"]] += 1
#     instruction = '''exitems in Context present within backticks 
# {}
# If the information isn't present, use "unknown"\nMake your response as short as possible in a list format.'''.format("\n".join([ f"- {field_name}"for field_name in field_name_to_count.keys()]))
    # instruction = ("\n".join([ f"Extract {field_name} from input within ``` and return <NA> if not found"for field_name in field_name_to_count.keys()]))
    instruction = "Give answers to following questions only. Add <NA> if value is not present\n" +  ("\n".join([ f"Extract value of {field_name} mentioned in the context."for field_name in field_name_to_count.keys()]))
    # instruction = "Extract value of following fields if the field is present in the context.\n" +  ("\n".join([ f"{field_name}"for field_name in field_name_to_count.keys()]))

    print(instruction)


df = pd.DataFrame({
    "page_id": [res.page_id for res in results],
    "field_name": [res.field_name for res in results],
    "ground_truth": [res.ground_truth for res in results],
    "prediction": [res.prediction for res in results],
    "llm_name": [res.llm_name for res in results],
})

print(f"Total results={len(results)}. Average Time Taken ={sum(times)/ len(times)}")
df.to_csv("results.csv", index=False)

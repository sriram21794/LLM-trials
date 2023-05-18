import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

# model_path = 'openlm-research/open_llama_3b_350bt_preview'
model_path = 'openlm-research/open_llama_7b_400bt_preview'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto'
)
raw_text = "odigiou\nPETERSEN PUBLISHING COMPANY\n1310702\nSOLD TO LORILLARD MEDIA SERVICES\nC/O BETTY EPPERSON\nbbb FIFTH AVE\nNEW YORK\nACCOUNT LORILLARD CORP DIV\n3012306\nSALESMAN\nBARTON\nCOLORATION: 4C\nPOSITION:\nC2\nPIE EAT\nNEWPORT\nPAG\nRATE ESTABLISHED BASED ON:\nAD TYPE: PETERSEN\nPUBLICATION\nHOT ROD\nDESCRIPTION\nestimate\n22,832.86\nNY 10103\nMAGAZINE\nBRAND\nISSUE\nHISERTION\nAD SIZE\nDISCOUNTS:\nPMN\nAGENCY COMMISSION\nNET SPACE\nY RO\nV DATE\nAllsvir\nPAD\nDATE\nCHECK\n(EXPLANATION OF BILLING)\nAN\nCONTRACT YEAR, FROM 03/87 THRU 12/87\nCONTRACT ORDER\nMAGAZINE NET\n2.\n\"^\ndaiB\nINDO\n3-1-87\n167717\n2-11-87\n7692307\nAS EARNED X\nRETURN DUPLICATE WITH REMITTANCE\n• TERMS: NET 30 DAYS\n2% ON NET SPACE CHARGE ONLY IF PAID\nWITHIN 10 DAYS.\nISSUE\nDATE\nINVOICE NO. CLIENT AUTHORIZATION\nORDER NO.\n0051 MAR-87 02/11/87 162212 1012348R\n2/84/87\n#3401\nPLEASE REMIT TO:\nPETERSEN PUBLISHING CO.\nP.O. BOX 2319\nNEW YORK, NY\nRATE CARD NO.: 054 Z\nOK\nSm\n2/34/87\nLINES\n1 PG\n6.0 %\nAMOUNT\nINVOICE\nFREQUENCY:\nNOTICE: IF YOU HAVE PREVIOUSLY PAID FOR THIS INSERTION PLEASE DISREGARD\nTHIS INVOICE. IT HAS BEEN PREPARED FOR YOUR RECORDS ONLY.\nCUSTOMER'S COPY\n10257-2319\nAMOUNTA\n27.702.00\n1662.12\n3,905.98\n22.133.90\n44.68\n21691.22/\n22.133.90,\n37065078"
prompt = f'Context: {raw_text}\nPrompt: Extract the Invoice number mentioned in the Context. Dont inlude any other text in output\nAnswer:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=16
)
print(tokenizer.decode(generation_output[0]))
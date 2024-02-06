from transformers import AutoTokenizer
import transformers
import torch


model = "meta-llama/Llama-2-7b-chat-hf"

#tokenizer = AutoTokenizer.from_pretrained(model, token=True)
tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    do_sample=False,
    top_k=1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200
)


from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

template='''[INST] <>
Only tell me the product names. The answer should only include ten names.
<>

{prompt}[/INST]'''

prompt_template = PromptTemplate(template=template, input_variables=["prompt"])

llm = HuggingFacePipeline(pipeline=pipeline)

llm_chain = LLMChain(prompt=prompt_template, llm=llm)


import pandas as pd
products = pd.read_csv('amazon_co-ecommerce_sample.csv', usecols=['product_name'])

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={'device': 'cpu'})


product_names = products['name'].values.astype(str)
product_embeddings = FAISS.from_texts(product_names, embeddings)



raw_query = 'what are the best gifts for boys under 5'
enhanced_query = llm_chain.run(raw_query)
product_embeddings.similarity_search_with_score(enhanced_query, k=10)
print(enhanced_query)
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import pandas as pd
import numpy as np
from transformers import pipeline
from nltk.translate.bleu_score import sentence_bleu
from datetime import datetime
import os
import pandas as pd
from rouge_score import rouge_scorer
import subprocess
result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value
        
# 初始化 RougeScorer
scorer = rouge_scorer.RougeScorer(
    ["rougeL"],  # 选择计算的ROUGE分数类型
    use_stemmer=True  # 是否使用词干提取器，默认为True
)

os.environ["HF_ENDPOINT"] = "huggingface.co"  # 使用镜像站点
CURRENT_PATH = os.getcwd() # 获取当前目录
CODEBERT_BASE_PATH = "microsoft/codebert-base" # codebert-base
#VICUNA_7B_PATH = "lmsys/vicuna-7b-v1.5" # vicuna-7b-v1.5
deepseek_v3_path = "deepseek-ai/DeepSeek-V3-Base" # deepseek-7b-v1.5
LLaMA_V2_13B_PATH = "lmsys/llama-v2-13b-v1.5" # llama-v2-13b-v1.5
gpt2 = "openai-community/gpt2"
# 下载并加载模型
tokenizer_codebert = AutoTokenizer.from_pretrained(CODEBERT_BASE_PATH)
model_codebert = AutoModel.from_pretrained(CODEBERT_BASE_PATH)

tokenizer_vicuna = AutoTokenizer.from_pretrained(gpt2)
model_vicuna = AutoModelForCausalLM.from_pretrained(gpt2, trust_remote_code=True)

EMBEDDING_MODEL_PATH = CODEBERT_BASE_PATH # 选择预训练模型的路径code_bert
# GENERATION_MODEL_PATH = VICUNA_7B_PATH # 选择生成模型的路径vicuna_7b
#GENERATION_MODEL_PATH = LLaMA_V2_13B_PATH # 选择生成模型的路径llama_13b
GENERATION_MODEL_PATH = gpt2 # 选择生成模型的路径deepseek_v3

EMBEDDING_MAX_LENGTH = 512 # 代码的最大长度
GENERATION_MAX_LENGTH = 512 # 生成的最大长度
TOP_K = 3 # 最相似的文档数
TASK_TYPE = "t03_review_comments_generation" # 任务类型

# 临时解决 OpenMP 错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

generator_time_list = []

# 记录程序开始时间
start_time = datetime.now()

def read_excel(file_path):
    # 使用 pandas 读取 Excel 文件
    df = pd.read_excel(file_path)  # 默认读取第一个工作表
    # 打印列名以检查
    print("Columns in Excel:", df.columns.tolist())

    # 检查列是否存在
    if 'diff_hunk' not in df.columns or 'body' not in df.columns:
        raise ValueError("Excel 文件中必须包含 'diff_hunk' 和 'body' 两列。")
    
    documents = []
    
    # 假设 Excel 文件包含两列：代码 (code) 和数据增强后的评审意见 (review_comment)
    for _, row in df.iterrows():
        documents.append({
            'diff_hunk': row['diff_hunk'],  # 代码列
            'body': row['body'],  # 评审意见列
        })
    
    return documents


def read_excel_to_diff_hunk_list(file_path):

    df = pd.read_excel(file_path)

    # 检查 'diff_hunk' 列是否存在
    if 'diff_hunk' not in df.columns:
        raise ValueError("Excel 文件中必须包含 'diff_hunk' 列。")

    # 将 'diff_hunk' 列的数据存储到列表中
    diff_hunk_list = df['diff_hunk'].tolist()

    return diff_hunk_list

def embed_text(text):
    # 确保输出包含 hidden_states
    print("embed_text_1_输入text是doc[patch]")
    # "pt" 表示返回 PyTorch (pt) 张量格式。如果你使用的是 TensorFlow，可以设置为 "tf"，这将返回 TensorFlow 张量。
    # 这个参数表示是否对输入文本进行截断，如果文本长度超过模型的最大输入长度（通常是 512 或其他最大值），它会将文本截断到最大长度。
    # 这个参数决定是否对输入文本进行填充（padding）。如果设置为 True，tokenizer 会将输入文本填充到指定的最大长度（通过 max_length 参数设置）。
    # 这个参数设置了最大输入长度。如果文本超过这个长度，会触发截断（truncation）；如果文本长度较短，会触发填充（padding）。
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=EMBEDDING_MAX_LENGTH)
    # inputs包含input_ids 和 attention_mask 两个tensor

    print("embed_text_2")
    # torch.no_grad() 是 PyTorch 中的一个上下文管理器（context manager），它用于禁用梯度计算，确保在此上下文中的所有操作都不会记录梯度，减少内存消耗，加速推理过程。
    with torch.no_grad():
        # **inputs 是一个解包字典的语法，表示将 inputs 字典中的所有键值对作为关键字参数传递给 auto_model。
        # 这个参数告诉模型在推理时，除了返回最终的预测结果外，还要返回所有层的隐藏状态（hidden states）。
        # 隐藏状态是模型内部每一层的输出，通常用于获取模型的中间表示或进行更复杂的操作（如特征提取、模型调优等）。
        outputs = auto_model(**inputs, output_hidden_states=True)  # 设置 output_hidden_states=True
    # 取最后一层的隐藏状态并对 token 级别的嵌入取平均
    print("embed_text_3")
    hidden_states = outputs.hidden_states  # 这是一个元组，包含所有层的输出
    # Layer 11 shape: torch.Size([1, 213, 768])
    # Layer 12 shape: torch.Size([1, 213, 768])
    # for i, layer in enumerate(hidden_states):
    #     print(f"Layer {i} shape: {layer.shape}")
    print("embed_text_4")
    last_hidden_state = hidden_states[-1]  # 获取最后一层的隐藏状态
    # print("last_hidden_state.shape : ", last_hidden_state.shape)
    # last_hidden_state.shape :  torch.Size([1, 213, 768])
    print("embed_text_5")
    # 对 token 级别的嵌入取平均
    # 对模型的最后一层隐藏状态 last_hidden_state 按照 token 维度（即 dim=1）进行 平均池化（mean pooling），从而得到每个输入序列的 文本级嵌入（text embedding）。
    # 具体而言，它将每个输入文本的所有 token 的隐藏状态取平均，从而得到一个固定长度的向量表示，用于后续的任务（如分类、相似度计算等）。
    # 作用是对每个文本的所有 token 的隐藏状态进行平均池化，从而生成每个文本的固定长度的嵌入表示（hidden_size 维度）。
    embeddings = last_hidden_state.mean(dim=1)  # 对每个 token 的嵌入取平均
    # print("embeddings.shape : ", embeddings.shape)
    # embeddings.shape :  torch.Size([1, 768])
    print("embed_text_6")
    # print("embeddings.numpy().flatten() : ", embeddings.numpy().flatten().shape)
    # embeddings.numpy().flatten().shape :  (768,)
    # 这行代码的作用是将 embeddings 张量（Tensor）转换为一个 NumPy 数组，然后 展平（flatten） 为一个一维数组。
    return embeddings.numpy().flatten()


def retrieve_similar_documents(input_content, k=TOP_K):
    # 先对输入进行嵌入，生成768维度的向量，方便和数据库中的向量进行对比
    input_embedding = embed_text(input_content).reshape(1, -1).astype(np.float32)
    # 这行代码使用已经训练好的 Faiss 索引 db_index 来查找与 input_embedding 最相似的 k 个文档的索引和距离。
    # distances：一个包含与 input_embedding 相似的文档的 距离 的数组。这个距离通常是 L2 距离（欧几里得距离）或余弦距离，具体取决于索引的构建方式。
    # indices：一个数组，包含最相似的文档在索引中的位置（索引值）。即，每个值表示一个文档的 ID 或位置。
    distances, indices = db_index.search(input_embedding, k)  # k是返回的最相似文档数
    # 这行代码返回最相似文档的 索引列表。
    return indices[0]  # 返回最相似的文档索引

def task03_generate_review(input_content, similar_docs):
    # 构建提示，包含输入代码和最相似的文档
    context = "\n".join([doc["body"] for doc in similar_docs])  # 获取相似文档的评论
    prompt = f"Input code change: {input_content}\n\nSimilar code review comments:\n{context}\n\nGenerate a new code review comment for the input change.\n"
    # 这一行代码使用先前定义的 generator（可能是一个基于 GPT 的文本生成模型）来生成新的代码评审评论。
    # generator(prompt) 返回一个包含生成文本的列表。列表中的第一个元素（即 [0]）是生成的文本。每个生成的文本都存储在字典中，字典的键是 'generated_text'，表示生成的内容。
    print("before generator")
    start_gen_time = datetime.now()
    review_comment = generator(prompt, max_length=GENERATION_MAX_LENGTH)[0]['generated_text']
    end_gen_time = datetime.now()
    elapsed_gen_time = end_gen_time - start_gen_time
    generator_time_list.append(elapsed_gen_time)
    print("after generator")
    return review_comment


if TASK_TYPE == "t01_code_problem_detection":
    pass
elif TASK_TYPE == "t02_code_problem_localization":
    pass
elif TASK_TYPE == "t03_review_comments_generation":
    file_path = CURRENT_PATH + "/test_data.xlsx"
    documents = read_excel(file_path) 
    input_path = CURRENT_PATH + "/test_data.xlsx"
    input_documents = read_excel_to_diff_hunk_list(input_path) #将diff_hunk列存入列表
    index_file = "D:/dataset/data/Comment_Generation/Comment_Generation_index.index"

    # 检查并创建目录
    index_dir = os.path.dirname(index_file)
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)  # 创建目录
    # index_file = CURRENT_PATH + "\\data\\Comment_Generation\\Comment_Generation_index.index"
    user_input_field = "diff_hunk"
    doc_output_field = "body"
    # print("documents : ", documents)


# 加载预训练模型（例如 CodeBERT 或类似的模型）
# AutoTokenizer 是 Hugging Face Transformers 库中的一个类，用于自动加载适合给定模型的分词器（tokenizer）。
# AutoTokenizer 可以自动选择适合特定预训练模型的分词器类型，简化了加载过程，避免了手动指定分词器类型的麻烦。
# 分词器（tokenizer）是自然语言处理（NLP）任务中一个非常重要的组件。
# 它的作用是将文本转换为模型能够理解的数值输入（即token ID），并且也能将模型输出的ID转换回文本。
# 通常，分词器会根据预训练的模型进行定制，以便与模型的预训练时使用的数据格式保持一致。
# 变量 tokenizer 会被赋值为一个分词器对象，之后可以使用这个对象来对文本进行分词（tokenization），
# 即将原始文本转化为适合模型输入的 token 序列。
# 例如，分词器会将“Hello, how are you?”分成单独的词、子词或者字符，并为这些单元分配唯一的ID。
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
# AutoModel 是 Hugging Face Transformers 库中的另一个类，它可以自动加载给定预训练模型的基础模型（不包括特定任务的头部层，如分类、问答等）。
# 该类适用于加载任何类型的模型（例如：BERT、GPT、T5、CodeBERT等）并自动适配。AutoModel 会根据传入的路径或模型名称自动加载模型的权重。
# 变量 auto_model 会被赋值为一个模型对象，之后可以使用该模型进行推理、训练等操作。例如，模型可以用于对输入文本进行分类、生成、填空任务等。
auto_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH)

# tokenizer = LlamaTokenizer.from_pretrained(CURRENT_PATH + VICUNA_7B_PATH)
# auto_model = LlamaForCausalLM.from_pretrained(CURRENT_PATH + VICUNA_7B_PATH)


# 将文档转为嵌入向量
print("===================10")
# document_embeddings = [embed_text(str(doc[user_input_field])) for doc in documents]
document_embeddings = []  
for doc in documents:
    # doc[user_input_field]为doc["patch"], doc["old_hunk"]
    # 把每个输入进行一个嵌入，流程：先用tokenizer进行划分，之后用嵌入模型model生成嵌入，去除最后一个隐藏层，池化后展开，作为最终的嵌入的结果
    embedding = embed_text(str(doc['diff_hunk'])) # 将代码片段转换为嵌入向量
    # print("embedding.shape : ", embedding.shape) # 大小是768
    document_embeddings.append(embedding) # 将嵌入向量添加到列表中
print("===================20")



# 假设文档嵌入的维度是 embedding_dimension
embedding_dimension = 768
# 这行代码的作用是创建一个 Faiss 索引，用于高效地进行 最近邻搜索（Nearest Neighbor Search），特别是基于 L2 距离（Euclidean 距离）。
# faiss.IndexFlatL2：这是 Faiss 中的一种索引类型，IndexFlatL2 用于通过 L2 距离（欧几里得距离）来查找最近邻。L2 距离是两点间的欧几里得距离，常用于度量向量之间的相似度。
# IndexFlatL2 会创建一个基础的、基于 L2 距离的索引结构，用于存储和搜索向量。
# 创建一个维度为768的，采用L2 距离进行查找的索引结构
db_index = faiss.IndexFlatL2(embedding_dimension)  # 使用 L2 距离度量
# 将所有嵌入向量添加到索引
# 这行代码的作用是将 document_embeddings 列表中的嵌入（embedding）向量堆叠（垂直拼接）成一个 NumPy 数组，并将其类型转换为 np.float32。
# 这是一个常见的数据处理操作，用于准备模型输出的嵌入向量，以便将其用于进一步的计算或存储。
# np.vstack() 是一个 NumPy 函数，用于将多个数组按垂直方向（沿着第一个轴）堆叠在一起。
# 将 document_embeddings_np 数组的数据类型转换为 np.float32
# float32 占用的内存比 float64 少，尤其是在处理大量数据时，可以显著减少内存的占用。
# 最终，document_embeddings_np 是一个二维的 NumPy 数组，形状为 [num_documents, embedding_dimension]，并且所有数据元素的类型都是 float32。
# 相当于把所有的参考文本，转化为embedding_dimension维度的嵌入向量，方便后面的问句的嵌入来检索
document_embeddings_np = np.vstack(document_embeddings).astype(np.float32)
# 这行代码的作用是将已经准备好的 文档嵌入向量 document_embeddings_np 添加到 Faiss 索引 db_index 中，从而使得该索引能够用于高效的 最近邻搜索。
db_index.add(document_embeddings_np)


# 保存索引
# 这两行代码的作用是将 Faiss 索引对象 db_index 保存到磁盘，以及 从磁盘加载 一个已保存的 Faiss 索引对象。
# Faiss 提供了 write_index 和 read_index 函数，用于将索引持久化到磁盘并在后续的程序中重新加载。
faiss.write_index(db_index, index_file) #将一个 Faiss 索引（db_index）保存到指定的文件路径（index_file）中
print(f"索引已保存到 {index_file}")
# index = faiss.read_index(index_file_path)


start_index = 1000
end_index = 1003
len_index = end_index - start_index
sum_bleu_value = 0
sum_rouge_value = 0
sum_em_value = 0


for i in range(start_index, end_index):
    choosed_index = i
    print("choosed_index = ", choosed_index)
    # 取出一行patch或者old_hunk作为输入
    # input_content = input_documents[choosed_index][user_input_field] # 选择一个代码片段
    input_content = input_documents[choosed_index]
    # print("input_content = ", input_content)
    # 获取最相似的索引列表
    similar_docs_indices = retrieve_similar_documents(input_content) # 获取最相似的代码索引
    # print("C60 Most similar documents indices:", similar_docs_indices)
    # 基于索引，取出文档
    similar_docs = [documents[i] for i in similar_docs_indices]
    # print("C80 Most similar documents:", similar_docs)
    # 这行代码创建了一个 文本生成管道（pipeline），并通过预训练模型进行文本生成。它加载了 CodeBERT 模型，并配置为生成文本的任务。我们逐步分析每个部分。
    # 相当于构建好生成文本的生成器
    generator = pipeline("text-generation", model = GENERATION_MODEL_PATH, trust_remote_code=True)
    
    
    TASK_TYPE == "t03_review_comments_generation"
    # 生成代码评审意见
    review_comment = task03_generate_review(input_content, similar_docs)
    # print("C90 Generated Review Comment:", review_comment)
    result = review_comment
    is_use_promt_and_generated = 1
    if is_use_promt_and_generated == 0:
        # 通过分割字符串获取 "Generate a new code review comment for the input change." 后面的部分
        result = review_comment.split("Generate a new code review comment for the input change.", 1)
        # 获取分割后的第二部分（即要保留的文本）
        if len(result) > 1:
            result = result[1].strip()  # 使用strip()去除前后的空白字符
        else:
            result = review_comment  # 如果没有找到该部分，返回原本的

    print("result = ", result)
    bleu_value = sentence_bleu(result, documents[choosed_index][doc_output_field])
    scores = scorer.score(result, documents[choosed_index][doc_output_field])
    sum_bleu_value += bleu_value
    sum_rouge_value += scores['rougeL'].fmeasure
    print("【bleu】", bleu_value)
    print("【rouge】", scores)
    print("【avg_bleu】", sum_bleu_value / (choosed_index - start_index + 1))
    print("【avg_rouge】", sum_rouge_value / (choosed_index - start_index + 1))
    print("\n")


    # if __name__ == "__main__":
    #     main()
print("【LEN】", len_index) 
print("【TASK_TYPE】", TASK_TYPE)
print()
print("【SUM_BLEU】", sum_bleu_value)
print("【SUM_ROUGE】", sum_rouge_value)
print()
print("【AVG_BLEU】", sum_bleu_value / len_index)
print("【AVG_ROUGE】", sum_rouge_value / len_index)


end_time = datetime.now()
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time}")

print("generator_time_list : ", generator_time_list)

# 【bleu】 1.2315055565533986e-231
# 【rouge】 {'rougeL': Score(precision=1.0, recall=0.06007067137809187, fmeasure=0.11333333333333333)}
# 【avg_bleu】 1.2846511706995118e-231
# 【avg_rouge】 0.0987648194099807


# 【LEN】 3
# 【TASK_TYPE】 t03_review_comments_generation

# 【SUM_BLEU】 3.853953512098535e-231
# 【SUM_ROUGE】 0.2962944582299421

# 【AVG_BLEU】 1.2846511706995118e-231
# 【AVG_ROUGE】 0.0987648194099807
# 程序运行时间: 0:01:17.226782






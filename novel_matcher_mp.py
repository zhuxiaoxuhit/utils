import re
import time
import argparse 
import sys
import os
import math
import shutil
import glob
#import jieba
import numpy as np
#from simhash import Simhash, SimhashIndex
import chardet
import codecs
import multiprocessing as mp
from multiprocessing import Process


class TextMatcher:
    def __init__(self, original_text_file, asr_text_file):
        #print(original_text_file)
        #print(asr_text_file)
        #encoding = get_encoding(original_text_file)
        #with open(original_text_file, 'r' ,encoding=encoding, errors='ignore') as file1:
        #with codecs.open(original_text_file, 'r', encoding=encoding, errors='ignore') as file1:
        with codecs.open(original_text_file, 'r', encoding="utf-8", errors='ignore') as file1:
            lines1 = file1.readlines()
            lines1 = [line for line in lines1 if line.strip()]
        # 将文本合并成一行并保留原来的换行符号
        original_text = ''.join(lines1)
        if len(original_text) <= 20:
            original_text = "xxxxx\nxxxxxxxx"
        #print("===origin_text===", flush=True)
        #print(original_text)
        
        #encoding1 = get_encoding(asr_text_file)
        #with open(asr_text_file, 'r', encoding=encoding1, errors='ignore') as file2:
        #with codecs.open(original_text_file, 'r', encoding=encoding1, errors='ignore') as file2:
        with codecs.open(asr_text_file, 'r', encoding="utf-8", errors='ignore') as file2:
            lines2 = file2.readlines()
        asr_text = ''.join(lines2)
        if len(asr_text) <= 20:
            asr_text = "xxxxx\nxxxxxxxx"
        #print("===asr_text===", flush=True)
        #print(asr_text, flush=True)
        #print("\n\n", flush=True)
        
        self.original_text = original_text.rstrip('\n')
        self.original_text_list, self.original_nopunc_text_list = self.split_lines_and_remove_punctuation(self.original_text)
        self.lens_per_line = [len(s) for s in self.original_nopunc_text_list]
        self.asr_text = self.remove_punctuation(asr_text)
        self.len_asr_text = len(self.asr_text)
        
        # best match
        self.min_match_txt_coef = 0.8
        self.max_match_txt_coef = 1.2
        self.min_char = math.floor(self.len_asr_text * self.min_match_txt_coef) 
        self.max_char = math.ceil(self.len_asr_text * self.max_match_txt_coef) 

        #self.sh_asr = Simhash(self.asr_text)

    # 模糊匹配
    def find_vague_match(self):
        #split_original = self.split_text(self.original_nopunc_text, len(self.asr_text))
        split_original_nopunc_tuple_lst = self.split_list(self.lens_per_line, self.len_asr_text)
        #print('=====')
        #print(self.len_asr_text)
        #print('=====')
        #print(self.lens_per_line)
        #print('=====')
        #print(split_original_nopunc_tuple_lst)
       
        sum_nb_lines = 0
        best_match_index = 0 
        max_similarity = 0
        min_distance = 10000
        for i, indexes_tuple in enumerate(split_original_nopunc_tuple_lst):
            start, indexes_list = indexes_tuple
            len_indexes_list = len(indexes_list)
            combined_text = "".join(self.original_nopunc_text_list[sum_nb_lines: sum_nb_lines+len_indexes_list])
            #print('=====')
            #print(combined_text)
            sum_nb_lines += len_indexes_list
            #combined_text_similarity = self.calculate_similarity(combined_text)
            combined_text_similarity = self.calculate_similarity_set(combined_text)
            #combined_text_similarity = self.simhash_similarity(combined_text)
            #print(combined_text_similarity)
           
            if combined_text_similarity >= max_similarity:
                max_similarity = combined_text_similarity
                best_match_index = i
            
            #if combined_text_similarity < min_distance:
            #    min_distance = combined_text_similarity
            #    best_match_index = i
       
        best_match_start, best_match_lst = split_original_nopunc_tuple_lst[best_match_index]
        #print(self.original_text_list[best_match_start])
        best_match = self.original_text_list[ best_match_start: best_match_start+len(best_match_lst)]
       
        #print("=====max_similarity=====")
        #print(max_similarity)
       
        #return (best_match, best_match_lst)
        return (split_original_nopunc_tuple_lst, best_match_index)
    
    def find_best_match(self, split_original_nopunc_tuple_lst, best_match_index):
        n = len(split_original_nopunc_tuple_lst)
        #combined_text = "".join(self.original_nopunc_text_list[best_match_index][0]: self.original_nopunc_text_list[best_match_index][0]+len(self.original_nopunc_text_list[best_match_index][1]))
        #print("-------combined------------")
        #print(self.original_nopunc_text_list[ split_original_nopunc_tuple_lst[best_match_index][0] : split_original_nopunc_tuple_lst[best_match_index][0] + len(split_original_nopunc_tuple_lst[best_match_index][1])])
        if best_match_index >= n-1:
            pad_list = [split_original_nopunc_tuple_lst[n-2], split_original_nopunc_tuple_lst[n-1]]
        elif best_match_index <= 0:
            pad_list = [split_original_nopunc_tuple_lst[0], split_original_nopunc_tuple_lst[1]]
        else:
            pad_list = [split_original_nopunc_tuple_lst[best_match_index-1], split_original_nopunc_tuple_lst[best_match_index], split_original_nopunc_tuple_lst[best_match_index+1]]
        start_index = pad_list[0][0]
        end_index = pad_list[-1][0] + len(pad_list[-1][1])
        triple_index_lst = []
        for lst in pad_list:
            for i in lst[1]:
                triple_index_lst.append(i)
        all_candidates = self.calculate_accumulation_indices(triple_index_lst, self.min_char, self.max_char, start_index)
        if len(all_candidates) <=1:
            return ("bad_match", 0.000001)
        
        sum_nb_lines = 0
        best_match_index = 0
        max_similarity = 0
        min_distance= 10000
        for i, candidate_lst in enumerate(all_candidates):
            combined_text = "".join(self.original_nopunc_text_list[candidate_lst[0] : candidate_lst[-1]+1])
            #print('=====')
            #print(combined_text)
            combined_text_similarity = self.calculate_similarity_set(combined_text)
            #combined_text_similarity = self.simhash_similarity(combined_text)
            if combined_text_similarity >= max_similarity:
                max_similarity = combined_text_similarity
                best_match_index = i
        
            #if combined_text_similarity < min_distance:
            #    min_distance = combined_text_similarity
            #    best_match_index = i
        
        best_match_lst = all_candidates[best_match_index]
        #print(self.original_text_list[best_match_start])
        best_match = self.original_text_list[best_match_lst[0] : best_match_lst[-1]+1]
        
        
        #if min_distance >= 10:
        #    return None 
        
        #return ("\n".join(best_match), min_distance)
        return ("\n".join(best_match), max_similarity)
         
    
    # 以下标 i 遍历输入的 list 数组，从 list[i]开始做累加，如果累加结果∈(min_acc, max_acc)闭区间内，则把下标组成的 list append 到 res 中。
    def calculate_accumulation_indices(self, input_list, min_acc, max_acc,  start_index):  
        res = []
        i = 0 
        while i < len(input_list):
            acc = 0 
            indices = [i]  # 存储累加的下标
            acc += input_list[i]
            for j in range(i+1, len(input_list)):
                acc += input_list[j]
                if min_acc <= acc <= max_acc:
                    indices.append(j)  # 添加下一个下标
                    res.append(indices[:])  # 将累加满足条件的下标添加到结果中
                elif acc > max_acc:
                    break
                #indices.append(j+1)  # 添加下一个下标
            i += 1
        return [[element + start_index for element in row] for row in res] 


  
    def split_lines_and_remove_punctuation(self, text):
        lines = text.split('\n')
        result = []
        for line in lines:
            no_punct = re.sub(r'[^\w\s]+', '', line)
            result.append(no_punct.replace(" ", ""))
        return (lines, result) 
    
    def remove_punctuation(self, text):
        no_punct = re.sub(r'[^\w\s]+', '', text)
        return no_punct.replace(" ", "")

    def split_text(self, text, length):
        return [text[i:i+length] for i in range(0, len(text), length)]

    def split_list(self, listA, num_asr):
        result = []
        sublist = []
        total = 0
        start = 0
    
        for i,num in enumerate(listA):
            # 如果加上当前数字后的累加和大于等于 num * 2，将当前子列表加入结果列表，并重置子列表和累加的数字
            #if total >= num_asr * 2:
            if total >= num_asr * 1:
                result.append((start, sublist))
                start = i
                sublist = []
                total = 0
    
            sublist.append(num)
            total += num
    
        # 将最后的子列表加入结果列表
        if sublist:
            #result.append(len(sublist))
            result.append((start, sublist))
    
        return result
   
    # 文本相似度1：计算交集大小/并集大小
    def calculate_similarity_set(self, text):
        normalized_modified = self.asr_text.lower()
        normalized_original = text.lower()
        intersection = len(set(normalized_modified) & set(normalized_original))
        union = len(set(normalized_modified)) + len(set(normalized_original))
        if union == 0:
            union = 100000 
        return intersection / union
   
    # 文本相似度2：余弦相似度
   
    ## 文本相似度3：simhash + 汉明距离
    #def simhash_similarity(self, text):
    #    sh_text = Simhash(text)
    #    #sh2 = Simhash(self.asr_text)
    #    distance = self.sh_asr.distance(sh_text)
    #    return distance


    # 文本相似度4：编辑距离
    def calculate_similarity_edit_distance(self, text):
        # 分词
        words1 = list(jieba.cut(text))
        words2 = list(jieba.cut(self.asr_text))
        
        # 初始化距离矩阵
        matrix = [[i + j for j in range(len(words2) + 1)] for i in range(len(words1) + 1)] 
        
        # 计算距离矩阵
        for i in range(1, len(words1) + 1): 
            for j in range(1, len(words2) + 1): 
                if words1[i - 1] == words2[j - 1]: 
                    matrix[i][j] = matrix[i - 1][j - 1]
                else:
                    matrix[i][j] = min(matrix[i - 1][j - 1], matrix[i - 1][j], matrix[i][j - 1]) + 1 
        
        # 返回编辑距离
        return matrix[len(words1)][len(words2)]

    # 文本相似度5：拼音+汉字混合方案



    def find_original_location(self, matched_text):
        original_sentences = self.original_text.split('.')
        matching_location = ""
        for sentence in original_sentences:
            if sentence.strip() in matched_text:
                matching_location += sentence.strip() + "."
        return matching_location.strip()

def get_encoding(file_path):
    print(file_path)
    with open(file_path, 'rb') as f:
        result = chardet.detect(f)
        encoding = result['encoding']
        return encoding

    #with codecs.open(file_path, 'r', encoding=encoding, errors='ignore') as file:
    #    content = file.read()
    #    #print(content)
    #    return content

def get_folder_names(path):
    folder_names = next(os.walk(path))[1]
    return folder_names
 
# 文本相似度1：计算交集大小/并集大小
def calculate_similarity_set_title(text1, text2):
    normalized_modified = text1.lower()
    normalized_original = text2.lower()
    intersection = len(set(normalized_modified) & set(normalized_original))
    union = len(set(normalized_modified)) + len(set(normalized_original))
    if union == 0:
        union = 100000 
    return intersection / union

def fine_best_title(book_name, book_lst):
    best_sim = 0
    best_book = None
    if len(book_lst) == 0 :
        return None 
    for book in book_lst:
        book_short = book.split("/")[-1].split(".")[0].replace("书名:", "书名_").replace("书作者:", "书作者_").replace("主播名字:", "主播名字_").replace("总章节数量:", "总章节数量_")
        #book_short = book.split("/")[-1].split(".")[0]
        sim = calculate_similarity_set_title(book_name, book_short)
        if sim >= best_sim:
            best_sim = sim
            best_book = book
    if best_sim > 0.8: 
        return best_book
    else:
        return None
        

def process_one(args, book_name):
    
    #### VAGUE Match book name 
    #original_book_files = glob.glob(args.original_txt_dir + "/*.txt")
    #origin_book_file = fine_best_title(book_name, original_book_files)
    #if origin_book_file == None:
    #    return
 
    original_book_file = os.path.join(args.original_txt_dir, book_name + ".txt")
    #print(original_book_file)
    #original_book_file = original_book_file
    #print(original_book_file)
    if os.path.exists(original_book_file):
        asr_files_lst = glob.glob(args.asr_dir + "/" + book_name + "/*.txt")
        #print("=========")
        #print(asr_files_lst)
        sum_sim = 0
        sum_chapter = 0
        for asr_file in asr_files_lst:

            ### TRY - EXCEPT  VERSION ###
            try:
                basename = os.path.basename(asr_file)
                if os.path.getsize(asr_file) < 40:
                    continue
                res_dir = os.path.join(args.match_results_dir, book_name)
                os.makedirs(res_dir, exist_ok=True)
                shutil.copy(asr_file, res_dir)

                matcher = TextMatcher(original_book_file, asr_file)
                split_original_nopunc_tuple_lst, best_match_index = matcher.find_vague_match()
                res, min_distance = matcher.find_best_match(split_original_nopunc_tuple_lst, best_match_index)
                res_file = os.path.join(res_dir, basename.split('.txt')[0] + "_sim_" + str(round(min_distance, 2))) + ".txt"
                with open(res_file, 'w', encoding='utf-8') as file:
                    file.write(res)
                sum_sim += min_distance
                sum_chapter += 1

            except:
                print("error in novel match:" + str(asr_file))
        if sum_chapter!= 0: 
            print(sum_sim/sum_chapter) 
        
            #### NORMAL VERSION ###
            #basename = os.path.basename(asr_file)
            ##if os.path.getsize(asr_file) == 0:
            #if os.path.getsize(asr_file) < 20 * 2:
            #    continue
            #res_dir = os.path.join(args.match_results_dir, book_name)
            #os.makedirs(res_dir, exist_ok=True)
            #shutil.copy(asr_file, res_dir)

            #matcher = TextMatcher(original_book_file, asr_file)
            #split_original_nopunc_tuple_lst, best_match_index = matcher.find_vague_match()
            #res, min_distance = matcher.find_best_match(split_original_nopunc_tuple_lst, best_match_index)
            #res_file = os.path.join(res_dir, basename.split('.txt')[0] + "_sim_" + str(round(min_distance, 2))) + ".txt"
            #with open(res_file, 'w', encoding='utf-8') as file:
            #    file.write(res)


def ProcessTask(args, book_names_lst):
    for book_name in book_names_lst:
        process_one(args, book_name.strip())


def ProcessList(argsv, lines,  ntask=1):

    if ntask == 1:
        ProcessTask(argsv, lines)
    else:
        chunk = len(lines) // ntask + 1 
        tasklist = []
        for i in range(ntask):
            if i * chunk >= len(lines):
                break
            tasklist.append(lines[i * chunk : (i + 1) * chunk])
        ntask = len(tasklist)
        sys.stdout.write(f"total {ntask} actual parrallel tasks\n")
        sys.stdout.flush()

        subtasks = [mp.Process(target=ProcessTask, args=(argsv, tasklist[i],)) for i in range(ntask)]

        [p.start() for p in subtasks]  
        [p.join() for p in subtasks]


if __name__=="__main__":
    start_time = time.time()  # 统计时间
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--asr_dir', type=str, default="/mnt/petrelfs/zhuxiaoxu/code/fairseq/utils2/examples/mms/data_prep/text/ebooks3/")
    parser.add_argument('-l', '--asr_lst', type=str, default="/mnt/petrelfs/zhuxiaoxu/code/fairseq/utils2/examples/mms/data_prep/text/ebooks3/")
    parser.add_argument('-t', '--original_txt_dir', type=str, default="/mnt/petrelfs/zhuxiaoxu/code/fairseq/utils2/examples/mms/data_prep/text/check_1222/origin_txts")
    parser.add_argument('-r', '--match_results_dir', type=str, default="/mnt/petrelfs/zhuxiaoxu/code/fairseq/utils2/examples/mms/data_prep/text/tttt")
    parser.add_argument('-n', '--ntask', type=int, default=1)
    args = parser.parse_args()
  
    book_name_lst = open(args.asr_lst).readlines()
    #book_name_lst =get_folder_names(args.asr_dir)
    #print(book_name_lst)
    if len(book_name_lst) <=2:
        print("ERROR: NO EBOOKS")
    os.makedirs(args.match_results_dir, exist_ok=True)   
    ProcessList(args, book_name_lst, args.ntask) 
    print('time elapsed: ', time.time() - start_time, 's')

